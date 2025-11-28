"""
Few-shot Prompting 기반 의도 분류 시스템

LLM에 의도 목록과 예시를 제공하여 직접 의도를 분류하도록 하는 방식
하드코딩된 예시 대신 Few-shot Prompting을 활용
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from app.logs.logger import setup_logger
from app.core.llm import LLMProvider
from app.core.llm.prompt_manager import PromptManager
from app.utils.language_detector import detect_language
from app.config import settings

logger = setup_logger('fewshot_intent_classifier')


class IntentCategory(Enum):
    """의도 카테고리 열거형"""
    GENERAL_CHAT = "general_chat"
    DOCUMENT_LIST = "document_list"
    DETAILED_QUESTION = "detailed"
    FOLLOW_UP_SUMMARY = "follow_up_summary"


@dataclass
class FewShotIntentResult:
    """Few-shot 의도 분류 결과"""
    primary_intent: IntentCategory
    confidence_score: float
    reasoning: str
    classification_method: str = "fewshot_llm"
    reference_index: Optional[int] = None  # 대화 참조 인덱스 (follow_up_summary용)
    reference_type: Optional[str] = None  # 참조 유형 (first, last, nth, recent)


class FewShotIntentClassifier:
    """Few-shot Prompting 기반 의도 분류기"""

    def __init__(self):
        """Few-shot 의도 분류기 초기화"""
        self.llm_provider = LLMProvider()
        self.llm = self.llm_provider.get_or_create_llm(settings.INTENT_ANALYSIS_MODEL)
        
        # 프롬프트 매니저 초기화
        self.prompt_manager = PromptManager()
        
        logger.info("Few-shot 의도 분류기 초기화 완료")

    def _create_fewshot_prompt(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Few-shot 프롬프트 생성 (YAML 파일 사용)"""

        logger.info(f"🔧 Few-shot 프롬프트 생성 시작: '{query[:50]}...'")

        # 언어 자동 감지
        detected_language = detect_language(query)
        logger.info(f"🌐 감지된 언어: {detected_language}")

        # 대화 히스토리 포맷팅
        history_text = "없음"
        if conversation_history:
            recent = conversation_history[-3:]
            history_parts = []
            for msg in recent:
                user_msg = msg.get('user', '')
                assistant_msg = msg.get('assistant', '')
                if user_msg and assistant_msg:
                    history_parts.append(f"사용자: {user_msg}\n어시스턴트: {assistant_msg}")
            if history_parts:
                history_text = "\n\n".join(history_parts)
            logger.info(f"📚 대화 히스토리 로드: {len(conversation_history)}개 메시지 중 최근 {len(recent)}개 사용")

        # YAML 파일을 한 번만 로드하여 의도 예시와 프롬프트 템플릿을 모두 가져오기
        try:
            import yaml
            from pathlib import Path

            yaml_path = Path(__file__).parent / "prompts" / "fewshot_intent_analysis.yaml"
            logger.info(f"📁 YAML 파일 로드 중: {yaml_path}")

            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 언어별 섹션에서 데이터 로드
            if detected_language not in data:
                raise ValueError(f"언어 '{detected_language}' 섹션이 YAML 파일에 없습니다.")
            
            lang_data = data[detected_language]
            if "fewshot_intent_analysis" not in lang_data:
                raise ValueError(f"언어 '{detected_language}' 섹션에 'fewshot_intent_analysis' 키가 없습니다.")

            # 의도 예시 가져오기
            if "intent_categories" not in lang_data["fewshot_intent_analysis"]:
                raise ValueError(f"언어 '{detected_language}' 섹션에 'intent_categories' 키가 없습니다.")
            
            intent_categories = lang_data["fewshot_intent_analysis"]["intent_categories"]
            
            if not intent_categories:
                raise ValueError(f"언어 '{detected_language}'의 intent_categories가 비어있습니다.")

            logger.info(f"📊 [{detected_language}] 로드된 의도 카테고리 수: {len(intent_categories)}")
            logger.info(f"📋 의도 카테고리 목록: {list(intent_categories.keys())}")

            examples_text = []
            for intent_name, intent_info in intent_categories.items():
                description = intent_info.get("description", "")
                examples = intent_info.get("examples", [])

                logger.debug(f"📝 {intent_name}: {len(examples)}개 예시 - {description}")

                examples_list = "\n".join([f"- {example}" for example in examples])
                examples_text.append(f"**{intent_name}**: {description}\n{examples_list}")

            intent_examples = "\n\n".join(examples_text)
            logger.info(f"✅ 의도 예시 포맷팅 완료 (총 길이: {len(intent_examples)} 문자)")

            # 프롬프트 템플릿 가져오기
            if "analysis_prompt_template" not in lang_data["fewshot_intent_analysis"]:
                raise ValueError(f"언어 '{detected_language}' 섹션에 'analysis_prompt_template' 키가 없습니다.")
            
            prompt_template = lang_data["fewshot_intent_analysis"]["analysis_prompt_template"]
            
            if not prompt_template:
                raise ValueError(f"언어 '{detected_language}'의 analysis_prompt_template이 비어있습니다.")
            
            logger.info(f"📄 [{detected_language}] 프롬프트 템플릿 길이: {len(prompt_template)} 문자")


            prompt = prompt_template.format(
                intent_examples=intent_examples,
                conversation_history=history_text,
                query=query
            )

            logger.info(f"✅ Few-shot 프롬프트 생성 완료 (총 길이: {len(prompt)} 문자)")
            logger.info(f"📝 생성된 프롬프트 미리보기:\n{prompt[:200]}")
            
        except Exception as e:
            logger.error(f"❌ YAML 프롬프트 템플릿 로드 실패: {e}")
            raise e

        return prompt

    def classify_intent(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> FewShotIntentResult:
        """Few-shot 기반 의도 분류 (JSON 응답 방식)"""
        try:
            logger.info(f"🔍 Few-shot 의도 분류 시작: '{query}'")

            # Few-shot 프롬프트 생성
            prompt = self._create_fewshot_prompt(query, conversation_history)

            # LLM 호출
            logger.info("🤖 LLM 호출 시작...")
            response = self.llm.invoke(prompt)
            response_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            logger.info(f"📤 LLM 응답 수신 (길이: {len(response_text)})")

            # JSON 응답 파싱
            import json
            import re

            # JSON 형태의 응답에서 중괄호 부분 추출 (멀티라인 포함)
            json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_response = json.loads(json_str)

                    # JSON에서 의도, 신뢰도, 추론 근거, reference 정보 추출
                    intent_str = parsed_response.get("intent", "").lower().strip()
                    confidence_score = float(parsed_response.get("confidence", 0.5))
                    reasoning = parsed_response.get("reasoning", "LLM 응답 파싱 성공")
                    reference_index = parsed_response.get("reference_index")
                    reference_type = parsed_response.get("reference_type")

                    # reference_index가 문자열로 파싱된 경우 정수로 변환
                    if reference_index is not None and isinstance(reference_index, str):
                        try:
                            reference_index = int(reference_index)
                        except ValueError:
                            logger.warning(f"⚠️ reference_index 변환 실패: {reference_index}, None으로 설정")
                            reference_index = None

                    logger.info(f"✅ JSON 파싱 성공: intent={intent_str}, confidence={confidence_score:.3f}")
                    if reference_index is not None:
                        logger.info(f"📍 대화 참조 정보: index={reference_index}, type={reference_type}")

                    # 의도 매칭
                    primary_intent = None
                    for intent_category in IntentCategory:
                        if intent_category.value == intent_str:
                            primary_intent = intent_category
                            logger.info(f"🎯 의도 매칭 성공: {intent_category.value}")
                            break

                    if not primary_intent:
                        logger.warning(f"⚠️ 알 수 없는 의도: {intent_str}, general_chat으로 폴백")
                        primary_intent = IntentCategory.GENERAL_CHAT
                        confidence_score = max(0.3, confidence_score * 0.5)  # 신뢰도 하향

                    # 신뢰도 범위 검증
                    confidence_score = max(0.0, min(1.0, confidence_score))

                    logger.info(f"✅ Few-shot 분류 결과: {primary_intent.value} (신뢰도: {confidence_score:.3f})")
                    logger.info(f"💭 분석 근거: {reasoning}")

                    return FewShotIntentResult(
                        primary_intent=primary_intent,
                        confidence_score=confidence_score,
                        reasoning=reasoning,
                        reference_index=reference_index,
                        reference_type=reference_type
                    )

                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 실패: {e}, 응답: {json_str}")
                    # 폴백: 기존 방식으로 의도만 추출
                    return self._fallback_parse(response_text)
            else:
                logger.warning(f"JSON 형식이 아닌 응답: {response_text[:100]}...")
                # 폴백: 기존 방식으로 의도만 추출
                return self._fallback_parse(response_text)

        except Exception as e:
            logger.error(f"❌ Few-shot 의도 분류 실패: {e}")
            return FewShotIntentResult(
                primary_intent=IntentCategory.GENERAL_CHAT,
                confidence_score=0.5,
                reasoning=f"분류 실패: {str(e)}"
            )

    def _fallback_parse(self, response_text: str) -> FewShotIntentResult:
        """폴백: JSON이 아닌 응답에서 의도 추출"""
        logger.warning("⚠️ 폴백 파싱 모드 활성화")

        response_lower = response_text.lower()

        # 의도 파싱
        primary_intent = None
        for intent_category in IntentCategory:
            if intent_category.value in response_lower:
                primary_intent = intent_category
                logger.info(f"🎯 폴백 의도 매칭: {intent_category.value}")
                break

        if not primary_intent:
            logger.warning(f"❌ 폴백에서도 의도를 찾을 수 없음")
            primary_intent = IntentCategory.GENERAL_CHAT

        # 기본 신뢰도 (낮게 설정)
        confidence_score = 0.6
        reasoning = f"폴백 파싱: {response_text[:100]}"

        return FewShotIntentResult(
            primary_intent=primary_intent,
            confidence_score=confidence_score,
            reasoning=reasoning
        )

    def get_top_k_intents(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.0,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[tuple]:
        """상위 K개의 의도 후보 반환 (Few-shot 방식에서는 단일 결과만 반환)"""
        result = self.classify_intent(query, conversation_history)
        return [(result.primary_intent, result.confidence_score)]
