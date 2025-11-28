from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any
import time
from datetime import datetime

from app.config import settings
from app.logs.logger import setup_logger

logger = setup_logger('document_loader')

class DocumentLoader:
    """문서 로딩을 담당하는 클래스"""
    
    def __init__(self):
        """DocumentLoader 초기화"""
        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP
        
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # 문장 단위 분할 우선
            length_function=len,
            is_separator_regex=False
        )
    
    def load_documents_from_pages(self, pages: List[Any], user_id: str, book_id: int) -> List[Document]:
        """
        페이지별로 개별 Document 객체를 생성합니다.
        
        Args:
            pages: 페이지 리스트 (pageKey, text 포함)
            user_id: 사용자 ID
            book_id: 도서 ID
            
        Returns:
            List[Document]: 페이지별 Document 객체 리스트
        """
        try:
            documents = []
            
            for page in pages:
                # 페이지 정보 추출
                if hasattr(page, 'pageKey'):
                    page_key = page.pageKey
                    page_text = page.text
                elif isinstance(page, dict):
                    page_key = page.get('pageKey', 'unknown')
                    page_text = page.get('text', '')
                else:
                    continue
                
                # 빈 텍스트는 건너뛰기
                if not page_text or not page_text.strip():
                    logger.warning(f"📋 페이지 {page_key}의 텍스트가 비어있어 건너뜁니다.")
                    continue
                
                # 페이지별 메타데이터 생성 (업로드 시간 포함)
                current_time_iso = datetime.now().isoformat()
                page_metadata = {
                    "user_id": user_id,
                    "book_id": book_id,
                    "page_key": page_key,
                    "upload_timestamp": current_time_iso,  # ISO 8601 형식
                }
                
                # 페이지별 Document 생성
                document = Document(
                    page_content=page_text,
                    metadata=page_metadata
                )
                
                documents.append(document)
            
            logger.info(f"✅ book_id {book_id} Document 생성 완료 (총 {len(documents)}개 페이지)")
            return documents
            
        except Exception as e:
            logger.error(f"페이지별 문서 로드 중 오류: {str(e)}", exc_info=True)
            return []

    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """문서를 청크로 분할합니다. 페이지별로 개별 처리하여 내용 손실을 방지합니다."""
        try:            
            # 페이지별로 청크 분할
            all_chunks = []
            
            for doc in documents:
                # 단일 문서를 청크로 분할
                doc_chunks = self._text_splitter.split_documents([doc])
                
                # 각 청크에 원본 메타데이터 복사 (제목 정보 포함)
                for i, chunk in enumerate(doc_chunks):
                    chunk.metadata.update(doc.metadata)
                    # 순서 보장을 위한 메타데이터 추가
                    chunk.metadata["chunk_order"] = i  # 청크 내 순서
                    chunk.metadata["page_order"] = doc.metadata.get("page_key", 0)  # 페이지 순서
                    chunk.metadata["document_order"] = len(all_chunks)  # 전체 문서 내 순서
                
                all_chunks.extend(doc_chunks)
            
            logger.info(f"✅ 청크 분할 완료: 총 {len(all_chunks)}개 청크 생성")
            
            return all_chunks
                
        except Exception as e:
            logger.error(f"텍스트 분할 중 오류: {str(e)}")
            # 오류 발생 시 원본 문서를 그대로 반환
            if documents:
                logger.warning("오류 발생으로 원본 문서를 청크로 사용합니다.")
                return documents
            return []
