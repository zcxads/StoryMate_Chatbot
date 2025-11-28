"""
Multi-turn RAG System State Definition
"""
from typing import Dict, List, Any, Optional, TypedDict


class MultiturnRAGState(TypedDict):
    """Defines the state for the multi-turn RAG system."""
    user_id: str
    query: str
    model_name: str
    conversation_history: List[Dict[str, str]]
    intent: Optional[str]
    search_context: Optional[str]  # 검색 컨텍스트
    detected_language: Optional[str]  # 감지된 언어 코드 (ko, en, ja, zh)
    retrieved_documents: Optional[List[Any]]
    answer: Optional[str]
    character_genre: Optional[str]  # 캐릭터 장르 (SCIENCE, HISTORY, PHILOSOPHY, LITERATURE, ART, FICTION)
    error: Optional[str]  # 오류 메시지
    execution_time: Optional[float]  # 실행 시간 (초)
    reference_index: Optional[int]  # 대화 참조 인덱스 (follow_up_summary용)
    reference_type: Optional[str]  # 참조 유형 (first, last, nth, recent)
