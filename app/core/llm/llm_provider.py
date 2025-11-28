from typing import Optional, Any, Dict

from app.utils.model import ModelFactory
from app.logs.logger import setup_logger

logger = setup_logger('llm_provider')

class LLMProvider:
    """LLM 모델 관리를 담당하는 클래스"""
    
    def __init__(self):
        """LLMProvider 초기화"""
        self._model_cache: Dict[str, Any] = {}
    
    def get_or_create_llm(self, model_name: str) -> Optional[Any]:
        """모델 캐시에서 가져오거나 새로 생성합니다."""
        if model_name not in self._model_cache:
            llm = ModelFactory.create_llm(model_name)
            if llm:
                self._model_cache[model_name] = llm
            else:
                return None
        
        return self._model_cache[model_name] 