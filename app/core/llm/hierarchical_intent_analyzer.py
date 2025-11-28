from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from app.logs.logger import setup_logger
from app.core.llm.fewshot_intent_classifier import FewShotIntentClassifier

logger = setup_logger('hierarchical_intent_analyzer')

class IntentCategory(Enum):
    """의도 카테고리 열거형"""
    GENERAL_CHAT = "general_chat"
    DOCUMENT_LIST = "document_list"
    DETAILED_QUESTION = "detailed"
    FOLLOW_UP_SUMMARY = "follow_up_summary"

class IntentConfidence(Enum):
    """의도 신뢰도 레벨"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class IntentAnalysis:
    """의도 분석 결과"""
    primary_intent: IntentCategory
    secondary_intents: List[IntentCategory]
    confidence: IntentConfidence
    confidence_score: float
    keywords: List[str]
    reasoning: str
    context_factors: Dict[str, Any]
    reference_index: Optional[int] = None  # 대화 참조 인덱스 (follow_up_summary용)
    reference_type: Optional[str] = None  # 참조 유형 (first, last, nth, recent)

class HierarchicalIntentAnalyzer:
    """Few-shot LLM 기반 의도 분석 시스템

    Few-shot LLM 분류만 사용 (예시 기반 프롬프팅) - 200-500ms
    """

    def __init__(self):
        """의도 분석기 초기화"""
        logger.info("🚀 Few-shot 의도 분석 시스템 초기화 시작")

        # Few-shot 기반 분류기
        self.fewshot_classifier = FewShotIntentClassifier()

        logger.info("✅ Few-shot 의도 분석 시스템 초기화 완료")
        logger.info(f"  - FewShotIntentClassifier (Few-shot LLM)")

    def _determine_confidence_level(self, confidence_score: float) -> IntentConfidence:
        """신뢰도 레벨 결정"""
        if confidence_score >= 0.7:
            return IntentConfidence.HIGH
        elif confidence_score >= 0.5:
            return IntentConfidence.MEDIUM
        else:
            return IntentConfidence.LOW

    def analyze_intent_hierarchically(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysis:
        """
        Few-shot LLM 기반 의도 분석을 수행합니다.

        Args:
            query: 사용자 질문
            conversation_history: 대화 히스토리
            user_context: 사용자 컨텍스트

        Returns:
            IntentAnalysis: 의도 분석 결과
        """
        logger.info(f"🔍 Few-shot 의도 분석 시작: '{query}'")
        logger.info("=" * 70)

        # Few-shot LLM 분류
        logger.info("📊 Few-shot LLM 분류")

        fewshot_result = self.fewshot_classifier.classify_intent(query, conversation_history)
        fewshot_intent = fewshot_result.primary_intent
        fewshot_confidence = fewshot_result.confidence_score

        logger.info(f"  ├─ Few-shot 의도: {fewshot_intent.value}")
        logger.info(f"  ├─ Few-shot 신뢰도: {fewshot_confidence:.4f}")
        logger.info(f"  └─ reference_index: {fewshot_result.reference_index}, reference_type: {fewshot_result.reference_type}")

        # Few-shot 결과 반환
        logger.info(f"✅ Few-shot 분류 완료")
        logger.info("=" * 70)

        confidence_level = self._determine_confidence_level(fewshot_confidence)

        return IntentAnalysis(
            primary_intent=fewshot_intent,
            secondary_intents=[],
            confidence=confidence_level,
            confidence_score=fewshot_confidence,
            keywords=[],
            reasoning=fewshot_result.reasoning,
            context_factors={
                "conversation_length": len(conversation_history) if conversation_history else 0,
                "has_context_shift": False,
                "user_context_available": user_context is not None,
                "fewshot_confidence": fewshot_confidence,
                "classification_method": "fewshot_only",
                "llm_calls": 1
            },
            reference_index=fewshot_result.reference_index,
            reference_type=fewshot_result.reference_type
        )
