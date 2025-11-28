from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from app.config import settings
from app.logs.logger import setup_logger

logger = setup_logger('vector_store')

@dataclass
class CollectionInfo:
    """컬렉션 정보를 담는 데이터 클래스"""
    name: str
    user_id: str
    book_key: str = "default"


class VectorStoreError(Exception):
    """VectorStore 관련 예외"""
    pass


class VectorStore:
    """벡터 DB 관리를 담당하는 클래스"""
    
    def __init__(self, embeddings: OpenAIEmbeddings):
        """VectorStore 초기화"""
        self.embeddings = embeddings
        self._client: Optional[QdrantClient] = None
        self._collection_cache: Dict[str, bool] = {}  # 컬렉션 존재 여부 캐시
    
    def _get_client(self) -> QdrantClient:
        """공통 Qdrant 클라이언트를 반환합니다. (HTTP 모드로 수정)"""
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=settings.QDRANT_TIMEOUT,
                    prefer_grpc=False  # HTTP 모드 사용
                )
                logger.info("Qdrant 클라이언트 초기화 완료 (HTTP 모드)")
            except Exception as e:
                logger.error(f"Qdrant 클라이언트 초기화 실패: {e}")
        return self._client
    
    def get_user_collection_name(self, user_id: str) -> str:
        """사용자별 컬렉션 이름을 생성합니다."""
        return f"{user_id}{settings.USER_COLLECTION_SUFFIX}"
    
    def get_user_collection(self, user_id: str) -> Optional[str]:
        """사용자별 컬렉션 목록을 반환합니다. (최적화된 버전)"""
        try:
            # 사용자별 컬렉션 이름 직접 생성
            expected_collection_name = self.get_user_collection_name(user_id)
            
            # 해당 컬렉션만 존재 여부 확인
            if self._collection_exists(expected_collection_name):
                return expected_collection_name
            else:
                logger.info(f"📝 사용자 {user_id}의 컬렉션이 없습니다 (새로 생성 예정): {expected_collection_name}")
                return None
            
        except Exception as e:
            logger.error(f"사용자 {user_id} 컬렉션 조회 중 예상치 못한 오류: {e}")
            raise VectorStoreError(f"컬렉션 조회 실패: {e}")
    
    def _collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부를 확인합니다. (캐시 무효화 개선)"""
        try:
            client = self._get_client()
            client.get_collection(collection_name)
            self._collection_cache[collection_name] = True
            return True
        except Exception as e:
            logger.info(f"컬렉션 {collection_name}이 존재하지 않습니다. 새로 생성 예정")
            self._collection_cache[collection_name] = False
            return False
    
    def _extract_collection_info(self, collection_name: str) -> CollectionInfo:
        """컬렉션 이름에서 정보를 추출합니다."""
        if not collection_name.endswith(settings.USER_COLLECTION_SUFFIX):
            raise ValueError(f"유효하지 않은 컬렉션 이름: {collection_name}")
        
        # suffix 제거
        name_without_suffix = collection_name[:-len(settings.USER_COLLECTION_SUFFIX)]
        
        # user_id_bookKey 형식인지 확인
        if "_" in name_without_suffix:
            parts = name_without_suffix.split("_", 1)
            user_id = parts[0]
            book_key = parts[1]
        else:
            # 기존 형식 (bookKey 없음)
            user_id = name_without_suffix
            book_key = "default"
        
        return CollectionInfo(name=collection_name, user_id=user_id, book_key=book_key)
    
    def _add_book_id_to_chunks(self, chunks: List[Document], book_id: int) -> None:
        """청크에 book_id 정보를 추가합니다."""
        for chunk in chunks:
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk.metadata['book_id'] = book_id
                # upload_timestamp가 이미 있으면 유지, 없으면 현재 시간으로 설정
                if 'upload_timestamp' not in chunk.metadata:
                    from datetime import datetime
                    import time
                    chunk.metadata['upload_timestamp'] = datetime.now().isoformat()
            
    def create_vector_db(self, chunks: List[Document], user_id: str, book_id: int) -> bool:
        """청크를 사용자별 벡터 DB에 저장합니다. (사용자 단위 컬렉션)"""
        try:
            collection_name = self.get_user_collection_name(user_id)
            logger.info(f"🔄 벡터 DB 생성 시작")
            logger.info(f"   👤 사용자: {user_id}")
            logger.info(f"   📚 도서 ID: {book_id}")
            logger.info(f"   🗂️  컬렉션: {collection_name}")
            logger.info(f"   📊 저장할 청크 수: {len(chunks)}개")
            
            # book_id 정보를 메타데이터에 추가
            self._add_book_id_to_chunks(chunks, book_id)
            
            if self._collection_exists(collection_name):
                logger.info(f"   📁 기존 컬렉션 {collection_name} 발견 - 청크 추가 모드")
                return self._add_to_existing_collection(collection_name, chunks, user_id)
            else:
                logger.info(f"   🆕 새 컬렉션 {collection_name} 생성 모드")
                return self._create_new_collection(collection_name, chunks)
                
        except Exception as e:
            logger.error(f"벡터 DB 생성 실패 - 사용자: {user_id}, 오류: {e}")
            raise VectorStoreError(f"벡터 DB 생성 실패: {e}")
    
    def _add_to_existing_collection(self, collection_name: str, chunks: List[Document], user_id: str) -> bool:
        """기존 컬렉션에 청크를 추가합니다."""
        logger.info(f"   📁 기존 컬렉션 {collection_name}에 청크 추가 시작")
        
        try:
            logger.info(f"   🔄 {len(chunks)}개 청크를 벡터 DB에 저장 중...")
            
            # 클라이언트 직접 사용하여 메타데이터 보장
            client = self._get_client()
            
            # 기존 포인트 수 확인
            existing_points = client.scroll(
                collection_name=collection_name,
                limit=10000,  # 충분히 큰 수로 모든 포인트 조회
                with_payload=False
            )[0]
            start_id = len(existing_points)
            
            # 각 청크를 개별적으로 추가하여 메타데이터 보장
            for i, chunk in enumerate(chunks):
                # 임베딩 생성
                embedding = self.embeddings.embed_query(chunk.page_content)
                
                # 메타데이터 준비
                payload = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                
                # 포인트 추가 (ID를 정수로 생성)
                client.upsert(
                    collection_name=collection_name,
                    points=[{
                        "id": start_id + i,  # 정수 ID 사용
                        "vector": embedding,
                        "payload": payload
                    }]
                )
            
            logger.info(f"   ✅ 기존 컬렉션 {collection_name}에 {len(chunks)}개 청크 추가 완료")
            return True
        except Exception as e:
            logger.error(f"   ❌ 기존 컬렉션에 청크 추가 실패: {e}")
            return False
    
    def _create_new_collection(self, collection_name: str, chunks: List[Document]) -> bool:
        """새 컬렉션을 생성합니다."""
        logger.info(f"   🆕 새 컬렉션 {collection_name} 생성 시작")
        
        try:
            logger.info(f"   🔄 {len(chunks)}개 청크로 새 컬렉션 생성 중...")
            
            # 클라이언트 직접 사용하여 메타데이터 보장
            client = self._get_client()
            
            # 컬렉션 생성 (이미 존재하면 무시)
            from qdrant_client.models import Distance, VectorParams
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding 크기
                        distance=Distance.COSINE
                    )
                )
            except Exception as e:
                # 컬렉션이 이미 존재하는 경우 무시
                logger.info(f"   📁 컬렉션 {collection_name}이 이미 존재합니다.")
            
            # 각 청크를 개별적으로 추가하여 메타데이터 보장
            for i, chunk in enumerate(chunks):
                # 임베딩 생성
                embedding = self.embeddings.embed_query(chunk.page_content)
                
                # 메타데이터 준비
                payload = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                
                # 포인트 추가 (ID를 정수로 생성)
                client.upsert(
                    collection_name=collection_name,
                    points=[{
                        "id": i,  # 정수 ID 사용
                        "vector": embedding,
                        "payload": payload
                    }]
                )
            
            logger.info(f"   ✅ 새 컬렉션 {collection_name} 생성 완료 - {len(chunks)}개 청크")
            return True
        except Exception as e:
            logger.error(f"   ❌ 새 컬렉션 생성 실패: {e}")
            return False
    
    def get_vector_db_for_user(self, user_id: str, collection_name: str) -> Optional[Qdrant]:
        """사용자별 벡터 DB를 반환합니다. (사용자 단위 컬렉션)"""
        try:
            if not collection_name:
                collection_name = self.get_user_collection_name(user_id)
            
            client = self._get_client()
            vector_db = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            return vector_db
            
        except Exception as e:
            logger.error(f"사용자 {user_id} 벡터 DB 조회 실패: {str(e)}")
            return None

    def get_user_document_detail(self, user_id: str) -> Optional[Dict[str, Any]]:
        """특정 사용자의 문서 상세 정보를 조회합니다. (사용자 단위 컬렉션)"""
        try:
            client = self._get_client()
            
            # 사용자의 컬렉션 조회
            collection_name = self.get_user_collection(user_id)
            
            if not collection_name:
                return None
            
            user_books = {}
            
            try:
                # 컬렉션의 모든 포인트 조회
                # 순서 보장을 위해 정렬 옵션 추가
                points = client.scroll(
                    collection_name=collection_name,
                    limit=10000,  # 대용량 문서 지원을 위해 증가
                    with_payload=True,
                    with_vectors=False
                )[0]
                
                logger.info(f"🔍 사용자 {user_id} 컬렉션 {collection_name}에서 {len(points)}개 포인트 조회")
                
                # 포인트를 순서 정보로 정렬
                sorted_points = self._sort_points_by_order(points)
                logger.info(f"📋 {len(sorted_points)}개 포인트 순서 정렬 완료")
                
                # 포인트별로 book_id 그룹화
                self._process_user_collection_points(sorted_points, user_books)
            
            except Exception as e:
                logger.error(f"컬렉션 {collection_name} 조회 실패: {e}")
            
            # 결과 형식 변환
            books_list = self._format_user_books_result(user_books)
            
            return {
                "user_id": user_id,
                "books": books_list,
                "total_books": len(books_list)
            }
            
        except Exception as e:
            logger.error(f"사용자 {user_id} 문서 상세 정보 조회 실패: {e}")
            raise VectorStoreError(f"문서 상세 정보 조회 실패: {e}")
    
    def _sort_points_by_order(self, points: List) -> List:
        """
        포인트를 순서 정보로 정렬합니다.
        
        Args:
            points: Qdrant에서 조회한 포인트 리스트
            
        Returns:
            List: 순서가 정렬된 포인트 리스트
        """
        try:
            def get_sort_key(point):
                """포인트의 정렬 키를 반환합니다."""
                if not point.payload or "metadata" not in point.payload:
                    return (0, 0, 0, 0, 0)  # 기본값
                
                metadata = point.payload["metadata"]
                book_id = metadata.get("book_id", 0)
                page_order = metadata.get("page_order", 0)
                chunk_order = metadata.get("chunk_order", 0)
                document_order = metadata.get("document_order", 0)
                upload_timestamp = metadata.get("upload_timestamp", 0)  # 업로드 시간 추가
                
                # book_id를 문자열로 통일하여 비교
                book_id_str = str(book_id) if book_id is not None else "0"
                
                # upload_timestamp를 안전하게 처리
                try:
                    upload_timestamp = int(upload_timestamp) if upload_timestamp is not None else 0
                except (ValueError, TypeError):
                    upload_timestamp = 0
                
                # 숫자와 문자열을 구분하여 정렬
                try:
                    # 숫자인 경우 숫자로 정렬
                    book_id_num = int(book_id_str)
                    book_id_key = (0, book_id_num)  # 숫자는 (0, 숫자)로 정렬
                except (ValueError, TypeError):
                    # 문자열인 경우 문자열로 정렬
                    book_id_key = (1, book_id_str)  # 문자열은 (1, 문자열)로 정렬
                
                return (book_id_key, upload_timestamp, page_order, chunk_order, document_order)
            
            # 순서 정보로 정렬
            sorted_points = sorted(points, key=get_sort_key)
            return sorted_points
            
        except Exception as e:
            logger.warning(f"포인트 정렬 중 오류 발생, 원본 순서 사용: {e}")
            return points

    def _process_user_collection_points(self, points: List, user_books: Dict[str, Any]) -> None:
        """사용자 컬렉션의 포인트들을 처리합니다."""
        logger.info(f"📋 총 {len(points)}개 포인트 처리 시작")
        
        for i, point in enumerate(points):
            if not point.payload:
                continue
            
            # book_id 추출 (메타데이터에서)
            book_id = self._extract_book_id_from_payload(point.payload)
            
            # book_id 정보 추출
            book_info = self._extract_page_info_from_payload(point.payload)
            
            if book_info:
                if book_id not in user_books:
                    user_books[book_id] = {"books": []}
                
                self._add_book_to_user_books(book_info, book_id, user_books)
        
        logger.info(f"✅ 포인트 처리 완료: {len(user_books)}개 book_id 그룹화됨")
    
    def _extract_book_id_from_payload(self, payload: Dict[str, Any]) -> str:
        """페이로드에서 book_id를 추출합니다."""
        if "metadata" in payload and "book_id" in payload["metadata"]:
            book_id = payload["metadata"]["book_id"]
            return str(book_id) if book_id is not None else "default"
        return "default"
    
    def _add_book_to_user_books(self, book_info: Any, book_id: str, user_books: Dict[str, Any]) -> None:
        """book_id 정보를 사용자 책에 추가합니다."""
        if isinstance(book_info, list):
            for book in book_info:
                if isinstance(book, dict):
                    if "book_id" not in book:
                        book["book_id"] = book_id
                    user_books[book_id]["books"].append(book)
        elif isinstance(book_info, dict):
            if "book_id" not in book_info:
                book_info["book_id"] = book_id
            user_books[book_id]["books"].append(book_info)
    
    def _format_user_books_result(self, user_books: Dict[str, Any]) -> List[Dict[str, Any]]:
        """사용자 책 정보를 결과 형식으로 변환합니다. 각 청크를 개별적으로 반환합니다."""
        books_list = []
        
        # book_id별로 순차 처리 (순서 보장)
        def sort_book_id_key(book_id):
            """book_id를 정렬 가능한 키로 변환합니다."""
            try:
                # 숫자인 경우 숫자로 정렬
                return (0, int(book_id))
            except (ValueError, TypeError):
                # 문자열인 경우 문자열로 정렬
                return (1, str(book_id))
        
        for book_id in sorted(user_books.keys(), key=sort_book_id_key):
            book_data = user_books[book_id]
            # book_id별로 모든 청크 정보 수집
            all_chunks = []
            
            # 순서 정보를 활용하여 정렬
            sorted_books = sorted(book_data["books"], 
                                key=lambda x: (
                                    x.get("page_keys", [0])[0] if x.get("page_keys") else 0,
                                    x.get("chunk_order", 0)
                                ))
            
            # 모든 책 정보를 처리하여 각 청크를 개별적으로 저장
            for book in sorted_books:
                page_keys = book.get("page_keys", [])
                content = book.get("content", "")
                
                # page_keys가 비어있거나 content가 비어있는 경우 처리
                if not page_keys:
                    page_keys = [1]
                
                if not content:
                    content = "문서 내용을 불러올 수 없습니다."
                
                # 각 page_key에 대해 개별 청크로 저장
                for page_key in page_keys:
                    chunk_info = {
                        "pageKey": page_key,
                        "text": content,
                        "upload_timestamp": book.get("upload_timestamp", 0)
                    }
                    all_chunks.append(chunk_info)
            
            # 청크들을 pageKey 기준으로 정렬 (이중 보장)
            all_chunks.sort(key=lambda x: x["pageKey"])
            
            books_list.append({
                "bookKey": str(book_id),
                "chunks": all_chunks
            })
        
        return books_list
    
    def _extract_page_info_from_payload(self, payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """페이로드에서 book_id 기준 정보를 추출합니다."""
        if "page_content" not in payload:
            return None
        
        page_content = payload["page_content"]
        
        # 메타데이터 추출
        metadata = payload.get("metadata", {})
        book_id = metadata.get("book_id", "default")
        
        # page_keys 처리 - 여러 형태 지원
        page_keys = metadata.get("page_keys", [])
        if not page_keys:
            # page_keys가 없으면 page_key(단수) 확인
            page_key = metadata.get("page_key")
            if page_key is not None:
                page_keys = [page_key]
            else:
                # 기본값으로 1 설정
                page_keys = [1]
        
        # 순서 정보 추출
        chunk_order = metadata.get("chunk_order", 0)
        page_order = metadata.get("page_order", 0)
        document_order = metadata.get("document_order", 0)
        
        # upload_timestamp 추출
        upload_timestamp = metadata.get("upload_timestamp", 0)
        
        # book_id 기준 정보 구성 (청크 정보 포함)
        book_info = {
            "book_id": book_id,
            "content": page_content,
            "page_keys": page_keys,
            "chunk_order": chunk_order,
            "page_order": page_order,
            "document_order": document_order,
            "upload_timestamp": upload_timestamp
        }
        return [book_info]
