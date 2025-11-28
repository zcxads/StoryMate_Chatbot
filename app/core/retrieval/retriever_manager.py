from typing import Optional, Any, Dict
from dataclasses import dataclass
from threading import Lock

from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever    
from qdrant_client import QdrantClient
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from app.config import settings
from app.logs.logger import setup_logger

logger = setup_logger('retriever_manager')

@dataclass
class SearchParams:
    """검색 파라미터를 담는 데이터 클래스"""
    k: int
    score_threshold: float
    timeout: int

class RetrieverManager:
    """검색기 관리를 담당하는 클래스"""
    
    def __init__(self, embeddings: OpenAIEmbeddings):
        """RetrieverManager 초기화"""
        self.embeddings = embeddings
        # 사용자별 BM25 캐시로 변경하여 동시성 문제 해결
        self._bm25_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()  # 캐시 접근 동기화
        
        # 공유 Qdrant 클라이언트 (연결 재사용)
        self._qdrant_client: Optional[QdrantClient] = None
        self._client_lock = Lock()
        
        # 검색 파라미터 설정
        self._search_params = SearchParams(
            k=settings.DEFAULT_SEARCH_K,
            score_threshold=settings.DEFAULT_SCORE_THRESHOLD,
            timeout=settings.DEFAULT_SEARCH_TIMEOUT
        )

    def _get_qdrant_client(self) -> QdrantClient:
        """공유 Qdrant 클라이언트를 반환합니다. (연결 재사용)"""
        if self._qdrant_client is None:
            with self._client_lock:
                if self._qdrant_client is None:
                    try:
                        self._qdrant_client = QdrantClient(
                            url=settings.QDRANT_URL,
                            api_key=settings.QDRANT_API_KEY,
                            timeout=settings.QDRANT_TIMEOUT,
                            prefer_grpc=False  # HTTP 모드 사용
                        )
                        logger.info("Qdrant 클라이언트 초기화 완료 (공유 연결)")
                    except Exception as e:
                        logger.error(f"Qdrant 클라이언트 초기화 실패: {e}")
                        raise
        return self._qdrant_client

    def get_search_params(self, search_params: SearchParams) -> SearchParams:
        """
        검색 파라미터를 반환합니다.
        
        Args:
            search_params: 검색 파라미터
            
        Returns:
            SearchParams: 검색 파라미터
        """
        # 기본 검색 파라미터 설정
        search_params = SearchParams(
            k=search_params.k,
            score_threshold=search_params.score_threshold,
            timeout=settings.MAX_SEARCH_TIME
        )
        
        logger.info(f"🔍 검색 파라미터: k={search_params.k}, threshold={search_params.score_threshold}")
        
        return search_params
    
    def create_retriever(self, vector_db, search_params: SearchParams) -> Optional[Any]:
        """
        기본 검색기를 생성합니다.
        
        Args:
            vector_db: 벡터 데이터베이스
            search_params: 검색 파라미터
            
        Returns:
            검색기 또는 None
        """
        try:
            # 기본 retriever 생성 (timeout 파라미터 제거)
            retriever = vector_db.as_retriever(
                search_kwargs={
                    "k": search_params.k,
                    "score_threshold": search_params.score_threshold
                }
            )
            
            return retriever
            
        except Exception as e:
            logger.error(f"❌ 검색기 생성 실패: {e}")
            return None
    
    def create_hybrid_retriever(self, vector_db, search_params: SearchParams, user_id: Optional[str] = None) -> Optional[Any]:
        """
        BM25Retriever와 벡터 Retriever를 결합한 EnsembleRetriever를 생성합니다.
        
        Args:
            vector_db: 벡터 데이터베이스
            search_params: 검색 파라미터
            user_id: 사용자 ID (캐시 분리를 위해)
            
        Returns:
            앙상블 검색기
        """
        try:
            # 1. 벡터 검색기 (시맨틱 검색) - timeout 파라미터 제거
            vector_retriever = vector_db.as_retriever(
                search_kwargs={
                    "k": search_params.k,
                    "score_threshold": search_params.score_threshold
                }
            )
            
            # 2. BM25 검색기 (사용자별 캐싱)
            collection_name = vector_db.collection_name
            user_cache_key = f"{user_id}_{collection_name}" if user_id else collection_name
            
            with self._cache_lock:
                if user_cache_key in self._bm25_cache:
                    bm25_retriever = self._bm25_cache[user_cache_key]
                    logger.info(f"✅ BM25 캐시 히트: {user_cache_key}")
                else:
                    bm25_retriever = self._create_bm25_retriever(vector_db, search_params, user_id)
                    if bm25_retriever:
                        self._bm25_cache[user_cache_key] = bm25_retriever
                        logger.info(f"✅ BM25 캐시 저장: {user_cache_key}")
            
            if bm25_retriever is None:
                logger.warning(f"⚠️ BM25 검색기 생성 실패, 벡터 검색기만 사용: {collection_name}")
                return vector_retriever
            
            # 3. 앙상블 검색기 생성 (벡터 + BM25)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[settings.ENSEMBLE_VECTOR_WEIGHT, settings.ENSEMBLE_BM25_WEIGHT]
            )
            
            logger.info(f"✅ 앙상블 검색기 생성 완료: {collection_name}")
            return ensemble_retriever
            
        except Exception as e:
            logger.error(f"❌ 앙상블 검색기 생성 실패: {e}")
            # 실패 시 기본 벡터 검색기 반환
            return vector_retriever
    
    def _create_bm25_retriever(self, vector_db, search_params: SearchParams, user_id: Optional[str] = None) -> Optional[Any]:
        """
        BM25 검색기를 생성합니다.
        
        Args:
            vector_db: 벡터 데이터베이스
            search_params: 검색 파라미터
            user_id: 사용자 ID (로깅용)
            
        Returns:
            BM25 검색기 또는 None
        """
        try:
            # 공유 Qdrant 클라이언트 사용
            client = self._get_qdrant_client()
            
            collection_name = vector_db.collection_name
            
            # 동시성 문제를 줄이기 위해 샘플링 크기 조정
            sample_size = min(settings.BM25_SAMPLE_SIZE, 500)  # 최대 500개로 제한
            
            logger.info(f"📊 BM25 검색기 생성 중: {collection_name} (샘플링: {sample_size}개)")
            
            # 샘플링된 문서만 가져오기 (Qdrant 클라이언트의 timeout 사용)
            points = client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not points:
                logger.warning(f"⚠️ BM25 검색기 생성 실패 (문서 없음): {collection_name}")
                return None
                        
            # Document 객체로 변환
            all_docs = []
            
            for point in points:
                page_content = point.payload.get("page_content", "")
                if not page_content.strip():
                    continue
                    
                # 메타데이터 추출
                metadata = {}
                if "metadata" in point.payload:
                    metadata = point.payload["metadata"]
                else:
                    metadata = {k: v for k, v in point.payload.items() if k != "page_content"}
                
                doc = Document(page_content=page_content, metadata=metadata)
                all_docs.append(doc)
            
            if not all_docs:
                logger.warning(f"⚠️ BM25 검색기 생성 실패 (유효한 문서 없음): {collection_name}")
                return None
            
            # BM25 검색기 생성 (timeout 파라미터 제거)
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = search_params.k
            
            logger.info(f"✅ BM25 검색기 생성 완료: {collection_name} (문서: {len(all_docs)}개)")
            return bm25_retriever
            
        except Exception as e:
            logger.error(f"❌ BM25 검색기 생성 실패: {e}")
            return None

    def clear_user_cache(self, user_id: str):
        """특정 사용자의 캐시를 정리합니다."""
        with self._cache_lock:
            keys_to_remove = [key for key in self._bm25_cache.keys() if key.startswith(f"{user_id}_")]
            for key in keys_to_remove:
                del self._bm25_cache[key]
            logger.info(f"🧹 사용자 {user_id} BM25 캐시 정리 완료")
