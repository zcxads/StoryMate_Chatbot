from app.logs.logger import setup_logger
from app.core.llm import LLMProvider
from app.core.llm.hierarchical_intent_analyzer import HierarchicalIntentAnalyzer
from app.config import settings

logger = setup_logger('intent_analyzer')

class IntentAnalyzer:
    """질문 의도 분석을 담당하는 클래스"""
    
    def __init__(self):
        """IntentAnalyzer 초기화"""
        self.llm_provider = LLMProvider()
        self.hierarchical_analyzer = HierarchicalIntentAnalyzer()
    
    def analyze_intent(self, query: str) -> str:
        """
        질문의 의도를 분석합니다.
        
        Args:
            query: 사용자 질문
            
        Returns:
            str: "document_list", "general_chat", "detailed", "follow_up_summary"
        """
        try:
            logger.info(f"🔍 의도 분석 시작: '{query}'")
            
            # 1차: 계층적 의도 분석 시스템 사용
            hierarchical_result = self.hierarchical_analyzer.analyze_intent_hierarchically(query)
            
            if hierarchical_result.confidence.value == "high":
                # 높은 신뢰도면 계층적 분석 결과 사용
                intent_result = hierarchical_result.primary_intent.value
                logger.info(f"📊 계층적 분석 결과 (높은 신뢰도): {intent_result} (신뢰도: {hierarchical_result.confidence_score:.2f})")
                logger.info(f"📝 분석 근거: {hierarchical_result.reasoning}")
                return intent_result
            else:
                # 낮은 신뢰도면 LLM으로 보완 분석
                logger.info(f"📊 계층적 분석 신뢰도 낮음 ({hierarchical_result.confidence_score:.2f}), LLM 보완 분석 수행")
                llm_analysis = self._analyze_intent_with_llm(query)
                
                # 계층적 분석과 LLM 분석 결과 비교 및 최종 결정
                if hierarchical_result.primary_intent.value == llm_analysis:
                    # 두 결과가 일치하면 사용
                    logger.info(f"📊 두 분석 결과 일치: {llm_analysis}")
                    return llm_analysis
                else:
                    # 불일치 시 계층적 분석 결과 우선 (더 정교한 분석)
                    logger.info(f"📊 분석 결과 불일치 - 계층적: {hierarchical_result.primary_intent.value}, LLM: {llm_analysis}")
                    logger.info(f"📊 계층적 분석 결과 채택: {hierarchical_result.primary_intent.value}")
                    return hierarchical_result.primary_intent.value
            
        except Exception as e:
            logger.warning(f"의도 분석 실패: {e}")
            return "general_chat"
    
    def _analyze_intent_with_llm(self, query: str) -> str:
        """
        LLM을 사용하여 질문의 의도를 분석합니다.
        """
        # 간단한 LLM 모델 사용 (빠른 응답을 위해)
        try:
            llm = self.llm_provider.get_or_create_llm(settings.INTENT_ANALYSIS_FALLBACK_MODEL)
        except:
            # 폴백 모델이 없으면 메인 모델 사용
            llm = self.llm_provider.get_or_create_llm(settings.INTENT_ANALYSIS_MODEL)
        
        # YAML 프롬프트 매니저에서 의도 분석 프롬프트 가져오기
        from app.core.llm.prompt_manager import PromptManager
        prompt_manager = PromptManager()
        intent_prompt = prompt_manager.get_intent_analysis_prompt(query)
        
        try:
            if llm is None:
                raise Exception("LLM 모델을 생성할 수 없습니다.")
                
            response = llm.invoke(intent_prompt)
            
            # AIMessage 객체를 문자열로 변환
            if hasattr(response, 'content'):
                result = response.content.strip().lower()
            elif hasattr(response, 'text'):
                result = response.text.strip().lower()
            else:
                result = str(response).strip().lower()
            
            # 응답 정규화
            if "document_list" in result:
                return "document_list"
            elif "detailed" in result:
                return "detailed"
            elif "follow_up_summary" in result:
                return "follow_up_summary"
            else:
                # 기본값
                return "general_chat"
                
        except Exception as e:
            logger.error(f"LLM 의도 분석 중 오류: {e}")
            raise e
        