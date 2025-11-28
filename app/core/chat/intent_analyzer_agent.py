from typing import Dict, List, Optional
from app.core.llm.hierarchical_intent_analyzer import HierarchicalIntentAnalyzer
from app.core.llm.prompt_manager import PromptManager
from app.states import MultiturnRAGState
from app.utils.language_detector import detect_language
from app.logs.logger import setup_logger

logger = setup_logger('intent_analyzer_agent')

class IntentAnalyzerAgent:
    """사용자 질문의 의도를 Few-shot LLM으로 분석하는 에이전트"""
    
    def __init__(self):
        """의도 분석 에이전트 초기화"""
        self.prompt_manager = PromptManager()
        # Few-shot LLM 의도 분석기 초기화
        self.hierarchical_analyzer = HierarchicalIntentAnalyzer()

    def analyze_intent(self, state: MultiturnRAGState) -> MultiturnRAGState:
        """사용자 질문의 의도를 Few-shot LLM으로 분석하여 intent와 search_context를 결정합니다."""
        query = state["query"]
        conversation_history = state["conversation_history"]

        # 1. 언어 자동 감지
        detected_language = detect_language(query)
        state["detected_language"] = detected_language
        logger.info(f"🌐 감지된 언어: {detected_language}")

        # 기본값 설정
        state["intent"] = "new_query"
        state["search_context"] = query

        try:
            # Few-shot LLM 의도 분석 실행
            logger.info("🔍 Few-shot LLM 의도 분석 시작...")
            hierarchical_result = self.hierarchical_analyzer.analyze_intent_hierarchically(
                query=query,
                conversation_history=conversation_history,
                user_context=None
            )

            # Few-shot 분석 결과를 의도로 설정
            state["intent"] = hierarchical_result.primary_intent.value
            state["reference_index"] = hierarchical_result.reference_index
            state["reference_type"] = hierarchical_result.reference_type

            logger.info(f"📊 Few-shot 분석 결과: {hierarchical_result.primary_intent.value}")
            logger.info(f"   └─ 신뢰도: {hierarchical_result.confidence_score:.3f} ({hierarchical_result.confidence.value})")
            logger.info(f"   └─ LLM 호출: {hierarchical_result.context_factors.get('llm_calls', 0)}회")
            logger.info(f"📝 분석 근거: {hierarchical_result.reasoning}")

            # reference 정보 로깅 (follow_up_summary인 경우에만)
            if hierarchical_result.reference_index is not None:
                logger.info(f"📍 대화 참조 정보: index={hierarchical_result.reference_index}, type={hierarchical_result.reference_type}")

            # search_context 설정 (의도에 따라)
            if hierarchical_result.primary_intent.value == "general_chat":
                # 일반 채팅: 문서 검색 불필요
                state["search_context"] = None
                logger.info("💬 일반 채팅 의도 → 문서 검색 생략")

            elif hierarchical_result.primary_intent.value == "follow_up_summary":
                # 직전 대화 요약: 문서 검색 불필요
                state["search_context"] = None
                logger.info("📋 직전 대화 요약 의도 → 문서 검색 생략")

            else:
                # 나머지 의도: 기본적으로 쿼리를 검색 컨텍스트로 사용
                state["search_context"] = query
                logger.info(f"🔎 검색 컨텍스트 설정: '{query}'")

        except Exception as e:
            logger.error(f"❌ Few-shot 의도 분석 중 오류 발생: {e}")
            # 최종 폴백: general_chat으로 처리
            logger.info("⚠️ 오류로 인한 최종 폴백: general_chat으로 처리")
            state["intent"] = "general_chat"
            state["search_context"] = None

        return state