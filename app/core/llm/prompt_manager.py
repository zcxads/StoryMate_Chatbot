import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from app.logs.logger import setup_logger

logger = setup_logger('prompt_manager')

class PromptManager:
    """YAML 파일 기반 프롬프트 매니저"""

    # 중복 로그 방지용 프로세스 단위 플래그
    _has_logged_intents: bool = False
    _has_logged_character_tones: bool = False

    def __init__(self):
        """YAML 프롬프트 매니저 초기화"""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.loaded_prompts = {}
        self._load_all_prompts()

        # 모든 언어의 의도 카테고리를 로드하고, 하위 호환을 위해 기본값을 유지
        self.intent_categories_by_language = self._load_all_intent_categories()
        # 하위 호환 필드: ko가 있으면 ko, 없으면 첫 가용 언어 사용
        self.intent_categories = (
            self.intent_categories_by_language.get("ko")
            or (next(iter(self.intent_categories_by_language.values())) if self.intent_categories_by_language else {})
        )
        self.character_tones = self._load_character_tones()
    
    def _load_all_prompts(self):
        """모든 YAML 프롬프트 파일을 로드합니다."""
        try:
            for yaml_file in self.prompts_dir.glob("*.yaml"):
                self._load_prompt_file(yaml_file)
        except Exception as e:
            logger.error(f"❌ YAML 프롬프트 로드 실패: {e}")

    def _load_intent_categories(self, language: str) -> Dict[str, Any]:
        """fewshot_intent_analysis.yaml에서 의도 카테고리를 동적으로 로드합니다.

        Args:
            language: 언어 코드 (ko, en, ja, zh)

        Returns:
            Dict[str, Any]: 의도 카테고리 딕셔너리
        """
        try:
            yaml_data = self.loaded_prompts.get("fewshot_intent_analysis")
            if not yaml_data:
                logger.warning("⚠️ fewshot_intent_analysis.yaml을 찾을 수 없습니다.")
                return {}

            # 새로운 구조 (언어별 섹션 존재)
            if language in yaml_data:
                lang_data = yaml_data[language]
                if "fewshot_intent_analysis" in lang_data and "intent_categories" in lang_data["fewshot_intent_analysis"]:
                    intent_categories = lang_data["fewshot_intent_analysis"]["intent_categories"]
                    logger.info(f"✅ [{language}] {len(intent_categories)}개 의도 카테고리 로드 완료")
                    return intent_categories

            # 구 구조 폴백 (하위 호환성)
            if "fewshot_intent_analysis" in yaml_data and "intent_categories" in yaml_data["fewshot_intent_analysis"]:
                intent_categories = yaml_data["fewshot_intent_analysis"]["intent_categories"]
                logger.info(f"✅ {len(intent_categories)}개 의도 카테고리 로드 완료 (구 구조)")
                return intent_categories

            logger.warning(f"⚠️ fewshot_intent_analysis.yaml에서 의도 카테고리를 찾을 수 없습니다. (언어: {language})")
            return {}
        except Exception as e:
            logger.error(f"❌ 의도 카테고리 로드 실패: {e}")
            return {}

    def _load_all_intent_categories(self) -> Dict[str, Dict[str, Any]]:
        """모든 언어(ko, en, ja, zh)의 의도 카테고리를 로드합니다.

        Returns:
            Dict[str, Dict[str, Any]]: 언어별 의도 카테고리 매핑
        """
        try:
            yaml_data = self.loaded_prompts.get("fewshot_intent_analysis")
            if not yaml_data:
                logger.warning("⚠️ fewshot_intent_analysis.yaml을 찾을 수 없습니다.")
                return {}

            languages = ["ko", "en", "ja", "zh"]
            by_language: Dict[str, Dict[str, Any]] = {}

            # 언어별 섹션
            for lang in languages:
                if isinstance(yaml_data, dict) and lang in yaml_data:
                    lang_data = yaml_data[lang]
                    if (
                        isinstance(lang_data, dict)
                        and "fewshot_intent_analysis" in lang_data
                        and isinstance(lang_data["fewshot_intent_analysis"], dict)
                        and "intent_categories" in lang_data["fewshot_intent_analysis"]
                    ):
                        by_language[lang] = lang_data["fewshot_intent_analysis"]["intent_categories"]
                        if not PromptManager._has_logged_intents:
                            logger.info(f"✅ [{lang}] {len(by_language[lang])}개 의도 카테고리 로드 완료")

            # 구 구조 폴백(루트에 존재) → 'default' 키로 저장
            if (
                "fewshot_intent_analysis" in yaml_data
                and isinstance(yaml_data["fewshot_intent_analysis"], dict)
                and "intent_categories" in yaml_data["fewshot_intent_analysis"]
            ):
                by_language.setdefault("default", yaml_data["fewshot_intent_analysis"]["intent_categories"])
                if not PromptManager._has_logged_intents:
                    logger.info(f"✅ default {len(by_language['default'])}개 의도 카테고리 로드 완료 (구 구조)")

            # 이 호출에서 처음으로 로그를 출력했다면, 이후부터는 중복 출력 방지
            if by_language and not PromptManager._has_logged_intents:
                PromptManager._has_logged_intents = True

            return by_language
        except Exception as e:
            logger.error(f"❌ 모든 언어 의도 카테고리 로드 실패: {e}")
            return {}

    def _load_character_tones(self) -> Dict[str, Any]:
        """character_tone.yaml에서 캐릭터 장르별 말투를 동적으로 로드합니다.

        Returns:
            Dict[str, Any]: 캐릭터 장르별 말투 딕셔너리
        """
        try:
            yaml_data = self.loaded_prompts.get("character_tone")
            if yaml_data and "character_tones" in yaml_data:
                character_tones = yaml_data["character_tones"]
                if not PromptManager._has_logged_character_tones:
                    logger.info(f"✅ {len(character_tones)}개 캐릭터 장르 말투 로드 완료")
                    PromptManager._has_logged_character_tones = True
                return character_tones
            else:
                logger.warning("⚠️ character_tone.yaml에서 캐릭터 말투를 찾을 수 없습니다.")
                return {}
        except Exception as e:
            logger.error(f"❌ 캐릭터 말투 로드 실패: {e}")
            return {}
    
    def _load_prompt_file(self, file_path: Path):
        """개별 YAML 프롬프트 파일을 로드합니다."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)
                file_name = file_path.stem
                self.loaded_prompts[file_name] = prompts
        except Exception as e:
            logger.error(f"❌ 프롬프트 파일 로드 실패 {file_path}: {e}")
    
    def get_prompt(self, file_name: str, prompt_key: str, language: str = None) -> Optional[str]:
        """
        특정 프롬프트를 가져옵니다.

        Args:
            file_name: YAML 파일 이름 (확장자 제외)
            prompt_key: 프롬프트 키 (점 표기법 지원)
            language: 언어 코드 (ko, en, ja, zh) - 지정하면 해당 언어 섹션에서 프롬프트 조회

        Returns:
            str: 프롬프트 텍스트
        """
        try:
            if file_name not in self.loaded_prompts:
                # 구 구조: 언어별 파일 확인 (하위 호환성)
                if language:
                    language_file_name = f"{file_name}_{language}"
                    if language_file_name in self.loaded_prompts:
                        file_name = language_file_name
                        logger.info(f"🌐 언어별 프롬프트 파일 사용: {file_name}")
                    else:
                        logger.error(f"❌ 프롬프트 파일을 찾을 수 없음: {file_name} 및 {language_file_name}")
                        return None
                else:
                    logger.error(f"❌ 프롬프트 파일을 찾을 수 없음: {file_name}")
                    return None

            prompts = self.loaded_prompts[file_name]

            # 새로운 구조: 언어 키가 최상위에 있는 경우 (ko, en, ja, zh)
            if language and language in prompts:
                logger.info(f"🌐 언어별 섹션 사용: {language}")
                prompts = prompts[language]
            elif language:
                # 언어가 지정되었지만 해당 언어 섹션이 없으면 구조 확인
                # 구 구조 (언어별 파일)이거나 언어 섹션이 없는 파일
                logger.info(f"ℹ️ 언어 섹션 없음, 기본 구조 사용 (언어: {language})")

            # 점 표기법으로 중첩된 키 접근
            keys = prompt_key.split('.')
            current = prompts

            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    logger.error(f"❌ 프롬프트 키를 찾을 수 없음: {prompt_key} (언어: {language})")
                    return None

            if isinstance(current, str):
                return current
            else:
                logger.error(f"❌ 프롬프트가 문자열이 아님: {prompt_key} (언어: {language})")
                return None

        except Exception as e:
            logger.error(f"❌ 프롬프트 가져오기 실패: {e}")
            return None
    
    def get_prompt_with_format(self, file_name: str, prompt_key: str, language: str = None, **kwargs) -> Optional[str]:
        """
        포맷팅된 프롬프트를 가져옵니다.

        Args:
            file_name: YAML 파일 이름
            prompt_key: 프롬프트 키
            language: 언어 코드 (ko, en, ja, zh)
            **kwargs: 포맷팅할 변수들

        Returns:
            str: 포맷팅된 프롬프트 텍스트
        """
        prompt = self.get_prompt(file_name, prompt_key, language=language)
        if prompt is None:
            return None

        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            logger.error(f"❌ 프롬프트 포맷팅 실패 - 누락된 변수: {e}")
            return prompt
        except Exception as e:
            logger.error(f"❌ 프롬프트 포맷팅 실패: {e}")
            return prompt
    
    def get_intent_analysis_prompt(self, query: str, conversation_history: Optional[list] = None) -> str:
        """
        의도 분석 프롬프트를 생성합니다.

        Args:
            query: 사용자 질문
            conversation_history: 대화 히스토리

        Returns:
            str: 의도 분석 프롬프트
        """
        # 대화 히스토리 포맷팅
        history_text = "없음"
        if conversation_history:
            history_parts = []
            for msg in conversation_history[-3:]:  # 최근 3개 메시지만
                user_msg = msg.get('user', '')
                assistant_msg = msg.get('assistant', '')
                if user_msg and assistant_msg:
                    history_parts.append(f"사용자: {user_msg}\n어시스턴트: {assistant_msg}")
            history_text = "\n\n".join(history_parts)

        # 의도 카테고리 포맷팅 (YAML에서 동적 로드된 데이터 사용)
        categories_text = []
        for category, info in self.intent_categories.items():
            desc = info.get('description', '')
            examples = info.get('examples', [])
            examples_text = ", ".join([f'"{ex}"' for ex in examples[:3]])  # 최대 3개 예시
            categories_text.append(f"**{category}**: {desc}\n예시: {examples_text}")
        intent_examples = "\n\n".join(categories_text)

        return self.get_prompt_with_format(
            "fewshot_intent_analysis",
            "fewshot_intent_analysis.analysis_prompt_template",
            query=query,
            conversation_history=history_text,
            intent_examples=intent_examples
        )

    def get_character_tone_instruction(self, character_genre: Optional[str] = None) -> str:
        """
        캐릭터 장르에 맞는 말투 지침을 반환합니다.

        Args:
            character_genre: 캐릭터 장르 (SCIENCE, HISTORY, PHILOSOPHY, LITERATURE, ART, FICTION)

        Returns:
            str: 말투 지침 프롬프트
        """
        try:
            # 장르가 지정되지 않은 경우 기본 말투 사용
            if not character_genre:
                default_tone = self.loaded_prompts.get("character_tone", {}).get("default_tone", {})
                return default_tone.get("instructions", "")

            # 장르에 해당하는 말투 가져오기
            if character_genre in self.character_tones:
                tone_data = self.character_tones[character_genre]
                tone_instruction = tone_data.get("tone_instructions", "")

                logger.info(f"🎭 캐릭터 장르 '{character_genre}' 말투 적용: {tone_data.get('name', '')}")
                return tone_instruction
            else:
                logger.warning(f"⚠️ 지원되지 않는 캐릭터 장르: {character_genre}")
                return ""

        except Exception as e:
            logger.error(f"❌ 캐릭터 말투 지침 가져오기 실패: {e}")
            return ""

 