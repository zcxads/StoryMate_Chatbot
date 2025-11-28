from typing import Dict, List, Any
from app.core.llm.prompt_manager import PromptManager
from app.logs.logger import setup_logger

logger = setup_logger('context_manager_agent')

class ContextManagerAgent:
    """대화 컨텍스트를 관리하는 에이전트"""
    
    def __init__(self, rag_system):
        """컨텍스트 관리 에이전트 초기화"""
        self.rag_system = rag_system
        self.prompt_manager = PromptManager()
    
    def manage_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """대화 컨텍스트를 관리하고 검색 전략을 결정합니다."""
        try:
            query = state["query"]
            conversation_history = state["conversation_history"]
            intent = state.get("intent", "general_chat")
            
            # 컨텍스트 초기화 여부 확인
            should_reset = self._should_reset_context(query, conversation_history)
            
            if should_reset:
                logger.info("🔄 컨텍스트 초기화 감지")
                # 새로운 주제를 위해 대화 히스토리 초기화
                state["conversation_history"] = []
            
            # 의도에 따라 검색 컨텍스트 결정
            state["search_context"] = "general_search"
            
        except Exception as e:
            logger.error(f"컨텍스트 관리 실패: {e}")
            state["search_context"] = "general_search"
        
        return state
    
    def _should_reset_context(self, query: str, conversation_history: List[Dict[str, str]]) -> bool:
        """대화 컨텍스트 초기화가 필요한지 판단합니다."""
        try:
            # 주제 전환을 나타내는 키워드
            reset_keywords = [
                "new topic", "different", "change subject", "let's talk about",
                "by the way", "speaking of", "on another note", "meanwhile"
            ]
            
            query_lower = query.lower()
            
            # 명시적 주제 전환 지표 확인
            for keyword in reset_keywords:
                if keyword in query_lower:
                    return True
            
            # 상당한 시간 간격이나 컨텍스트 전환 확인
            if len(conversation_history) > 10:
                # 대화가 매우 길면 초기화 고려
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"컨텍스트 초기화 확인 실패: {e}")
            return False 