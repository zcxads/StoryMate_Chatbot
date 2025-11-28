from typing import List, Dict
import random
import re
from datetime import datetime, timedelta
from dateutil import parser as date_parser

from app.logs.logger import setup_logger

logger = setup_logger('document_context')

class DocumentContextManager:
    """문서 컨텍스트 관리를 담당하는 클래스"""
    
    def __init__(self, rag_system):
        """DocumentContextManager 초기화"""
        self.rag_system = rag_system
    
    def get_all_user_documents_context(self, user_id: str, query: str = "", prompt_type: str = "") -> str:
        """
        사용자의 모든 문서 정보를 컨텍스트로 반환합니다.
        의도 타입에 따라 적절한 문서만 선택합니다.
        
        Args:
            user_id: 사용자 ID
            query: 사용자 질문
            prompt_type: 의도 타입 (document_list, document_order 등)
            
        Returns:
            str: 선택된 문서 정보를 포함한 컨텍스트
        """
        try:
            # 사용자의 모든 문서 정보 조회
            user_document_detail = self.rag_system.get_user_document_detail(user_id)
            
            if not user_document_detail or not user_document_detail.get("books"):
                return "사용자가 업로드한 문서가 없습니다."
            
            books = user_document_detail["books"]

            # 모든 문서 사용
            selected_books = books
            
            context_parts = []
            
            for book in selected_books:
                book_key = book.get("bookKey", "unknown")
                chunks = book.get("chunks", [])
                
                if chunks:
                    # 샘플링된 청크들만 사용하여 컨텍스트 구성
                    sampled_text = self._sample_chunks_for_context(chunks, max_chunks=3)
                    
                    if sampled_text.strip():
                        # 샘플링된 문서 내용을 컨텍스트에 추가
                        context_parts.append(f"{sampled_text.strip()}")
                    else:
                        context_parts.append(f"문서 {book_key}: 내용 없음")
                else:
                    context_parts.append(f"문서 {book_key}: 내용 없음")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"사용자 {user_id} 문서 정보 조회 실패: {e}")
            return "문서 정보를 가져올 수 없습니다."

    def _sample_chunks_for_context(self, chunks: List[Dict], max_chunks: int = 3) -> str:
        """
        청크들에서 샘플링하여 컨텍스트용 텍스트를 생성합니다.
        
        Args:
            chunks: 청크 리스트
            max_chunks: 샘플링할 최대 청크 수
            
        Returns:
            str: 샘플링된 텍스트
        """
        try:
            if not chunks:
                return ""
            
            # 청크 수가 max_chunks 이하면 모두 사용
            if len(chunks) <= max_chunks:
                sampled_chunks = chunks
            else:
                # 전략적 샘플링: 시작, 중간, 끝 부분에서 샘플링
                sampled_chunks = []
                
                # 1. 첫 번째 청크 (시작 부분)
                sampled_chunks.append(chunks[0])
                
                # 2. 중간 청크들 (문서의 핵심 내용)
                if len(chunks) > 2:
                    middle_start = len(chunks) // 3
                    middle_end = 2 * len(chunks) // 3
                    middle_chunks = chunks[middle_start:middle_end]
                    
                    # 중간 청크들에서 1-2개 샘플링
                    if len(middle_chunks) > 1:
                        # 중간 청크들 중에서 랜덤하게 선택
                        middle_sample = random.sample(middle_chunks, min(1, len(middle_chunks)))
                        sampled_chunks.extend(middle_sample)
                    else:
                        sampled_chunks.extend(middle_chunks)
                
                # 3. 마지막 청크 (끝 부분)
                if len(chunks) > 1:
                    sampled_chunks.append(chunks[-1])
                
                # max_chunks 개수로 제한
                sampled_chunks = sampled_chunks[:max_chunks]
            
            # 샘플링된 청크들의 텍스트 결합
            sampled_text = ""
            for chunk in sampled_chunks:
                text = chunk.get("text", "")
                if text:
                    sampled_text += text + " "
            
            return sampled_text.strip()
            
        except Exception as e:
            logger.warning(f"청크 샘플링 중 오류: {e}")
            # 오류 발생 시 첫 번째 청크만 사용
            if chunks:
                return chunks[0].get("text", "").strip()
            return ""
