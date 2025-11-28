from typing import Dict, Any
from langsmith import traceable
from app.core.llm.prompt_manager import PromptManager
from app.core.llm.llm_provider import LLMProvider
from app.logs.logger import setup_logger
from app.utils.language_detector import detect_language
from app.config import settings

logger = setup_logger('answer_generator_agent')


class AnswerGeneratorAgent:
    """답변을 생성하는 에이전트"""
    
    def __init__(self, rag_system):
        """답변 생성 에이전트 초기화"""
        self.rag_system = rag_system
        self.prompt_manager = PromptManager()
    
    async def generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """검색된 문서와 메모리를 바탕으로 답변을 생성합니다."""
        try:
            query = state["query"]
            intent = state.get("intent", "general_chat")
            retrieved_documents = state.get("retrieved_documents", []) or []
            character_genre = state.get("character_genre", None)  # 캐릭터 장르 가져오기

            # 언어 자동 감지
            detected_language = detect_language(query)
            logger.info(f"🌐 감지된 언어: {detected_language} - 쿼리: {query[:50]}...")

            # 캐릭터 장르 로그
            if character_genre:
                logger.info(f"🎭 캐릭터 장르: {character_genre}")

            # LLM 가져오기
            llm_provider = LLMProvider()
            llm = llm_provider.get_or_create_llm(settings.ANSWER_GENERATION_MODEL)

            logger.info(f" 의도별 프롬프트 사용: {intent}")

            # 일반 채팅: 문서 없이 바로 답변 생성
            if intent == "general_chat":
                logger.info("💬 일반 채팅 모드: 문서 기반 프롬프트 없이 직접 답변 생성")

                # 일반 채팅용 시스템 프롬프트 사용
                system_prompt = self.prompt_manager.get_prompt(
                    "multiturn_answer_generation",
                    "answer_generation.system_prompt",
                    language=detected_language
                )

                # 캐릭터 장르별 말투 추가
                character_tone = self.prompt_manager.get_character_tone_instruction(character_genre)
                if character_tone:
                    system_prompt = f"{system_prompt}\n\n{character_tone}"

                prompt_template = self.prompt_manager.get_prompt(
                    "multiturn_answer_generation",
                    "answer_generation.general_chat_prompt",
                    language=detected_language
                )
                task_prompt = prompt_template.format(query=query)
                prompt = f"{system_prompt}\n\n{task_prompt}"

                # LangSmith 추적용 컨텍스트 기록
                context_info = {
                    "system_prompt": system_prompt,
                    "task_prompt": task_prompt,
                    "retrieved_documents": "",
                    "conversation_history": "",
                    "detected_language": detected_language
                }
                await self._log_final_prompt(prompt, intent, context_info)

            else:
                retrieved_docs_parts = [doc.page_content for doc in retrieved_documents if getattr(doc, "page_content", "")]
                conversation_parts = []
                conversation_history = state.get("conversation_history") or []
                
                # 폴백 플래그 확인 (대화 기록 우선 사용 여부)
                fallback_to_memory = state.get("fallback_to_memory", False)
                memory_conversations = state.get("memory_conversations", [])
                
                # 폴백 모드: 대화 기록 우선 사용
                if fallback_to_memory and memory_conversations:
                    logger.info("폴백 모드: 대화 기록 우선 사용")
                    # 대화 기록에서 context 추출
                    memory_parts = []
                    for doc in memory_conversations[:5]:
                        if hasattr(doc, 'metadata'):
                            user_query = doc.metadata.get("query", "")
                            assistant_response = doc.metadata.get("response", "")
                            if user_query and assistant_response:
                                memory_parts.append(f"Q: {user_query}\nA: {assistant_response}\n")
                    
                    # 실제 문서가 없거나 적으면 대화 기록을 주 컨텍스트로 사용
                    if len(retrieved_docs_parts) < 3:
                        retrieved_docs_parts = ["\n".join(memory_parts)] if memory_parts else retrieved_docs_parts
                        logger.info(f"대화 기록을 주 컨텍스트로 사용 ({len(memory_parts)}개 대화)")
                    else:
                        # 실제 문서와 대화 기록 병합
                        conversation_parts.extend(memory_parts)
                        logger.info(f"실제 문서 + 대화 기록 병합 (문서: {len(retrieved_docs_parts)}개, 대화: {len(memory_parts)}개)")
                
                # 일반적인 대화 기록 추가
                if conversation_history:
                    for conv in conversation_history:
                        user_text = conv.get("user", "")
                        assistant_text = conv.get("assistant", "")
                        if user_text and assistant_text:
                            conversation_parts.append(f"Q: {user_text}\nA: {assistant_text}\n")
                
                # 검색 결과가 없는 경우 기본 응답 생성
                if not retrieved_docs_parts:
                    logger.info("📝 검색 결과가 없어 기본 응답을 생성합니다.")
                    answer_text = f"죄송합니다. '{query}'에 대한 정보를 찾을 수 없습니다."
                    state["answer"] = answer_text
                    
                    # 검색 결과가 없어도 대화 기록은 저장해야 함
                    try:
                        user_id = state["user_id"]
                        logger.info(f"💾 대화 기록 저장 시작 (검색 결과 없음) - 사용자: {user_id}")
                        logger.info(f"💾 저장할 질문: {query[:50]}... ({len(query)} 문자)")
                        logger.info(f"💾 저장할 답변: {answer_text[:50]}... ({len(answer_text)} 문자)")
                        
                        self.rag_system.chat_history_manager.add_to_chat_history(
                            user_id, query, answer_text
                        )
                        
                        logger.info("✅ 대화 기록 저장 완료 (검색 결과 없음)")
                    except Exception as save_error:
                        logger.error(f"❌ 대화 기록 저장 실패 (검색 결과 없음): {save_error}", exc_info=True)
                    
                    return state

                # 언어별 프롬프트 사용
                system_prompt = self.prompt_manager.get_prompt(
                    "multiturn_answer_generation",
                    "answer_generation.system_prompt",
                    language=detected_language
                )

                # 캐릭터 장르별 말투 추가
                character_tone = self.prompt_manager.get_character_tone_instruction(character_genre)
                if character_tone:
                    system_prompt = f"{system_prompt}\n\n{character_tone}"

                if intent == "document_list":
                    prompt_template = self.prompt_manager.get_prompt(
                        "multiturn_answer_generation",
                        "answer_generation.document_list_prompt",
                        language=detected_language
                    )
                    task_prompt = prompt_template.format(
                        retrieved_documents="\n\n".join(retrieved_docs_parts),
                        query=query
                    )
                elif intent == "follow_up_summary":
                    # 과거 대화 요약용 프롬프트
                    prompt_template = self.prompt_manager.get_prompt(
                        "multiturn_answer_generation",
                        "answer_generation.follow_up_summary_prompt",
                        language=detected_language
                    )

                    # reference_index를 사용하여 특정 대화 가져오기
                    reference_index = state.get("reference_index")
                    reference_type = state.get("reference_type")

                    referenced_conversation = ""
                    if reference_index is not None and conversation_history:
                        try:
                            # reference_index로 해당 대화 가져오기 (0=첫번째, -1=마지막)
                            if -len(conversation_history) <= reference_index < len(conversation_history):
                                conv = conversation_history[reference_index]
                                user_text = conv.get("user", "")
                                assistant_text = conv.get("assistant", "")
                                if user_text and assistant_text:
                                    referenced_conversation = f"Q: {user_text}\nA: {assistant_text}"
                                    logger.info(f"📍 참조된 대화: {user_text[:50]}...")
                            else:
                                logger.warning(f"⚠️ reference_index={reference_index}가 범위를 벗어남 (대화 수: {len(conversation_history)})")
                        except Exception as e:
                            logger.error(f"❌ reference_index 처리 중 오류: {e}")

                    # referenced_conversation이 없으면 전체 대화 히스토리 사용
                    if not referenced_conversation and conversation_parts:
                        referenced_conversation = "\n\n".join(conversation_parts)
                        logger.info(f"🔍 follow_up_summary - 전체 대화 히스토리 사용 ({len(conversation_parts)}개)")

                    if not referenced_conversation:
                        referenced_conversation = "이전 대화가 없습니다."
                        logger.warning("⚠️ 참조할 대화를 찾을 수 없음")

                    task_prompt = prompt_template.format(
                        conversation_history=referenced_conversation,
                        query=query
                    )
                else:
                    prompt_template = self.prompt_manager.get_prompt(
                        "multiturn_answer_generation",
                        "answer_generation.detailed_question_prompt",
                        language=detected_language
                    )
                    task_prompt = prompt_template.format(
                        retrieved_documents="\n\n".join(retrieved_docs_parts),
                        conversation_history="\n\n".join(conversation_parts) if conversation_parts else "이전 대화가 없습니다.",
                        query=query
                    )

                prompt = f"{system_prompt}\n\n{task_prompt}"

                context_info = {
                    "system_prompt": system_prompt,
                    "task_prompt": task_prompt,
                    "retrieved_documents": "\n\n".join(retrieved_docs_parts),
                    "conversation_history": "\n\n".join(conversation_parts) if conversation_parts else "",
                    "detected_language": detected_language
                }
                await self._log_final_prompt(prompt, intent, context_info)
            
            response = llm.invoke(prompt)
            
            # 응답 텍스트 추출 개선
            answer_text = None
            
            # LangChain ChatResult/ChatGeneration 처리
            if hasattr(response, 'content') and response.content:
                answer_text = str(response.content).strip()
                logger.info("✅ 응답 추출: response.content 사용")
            elif hasattr(response, 'text') and response.text:
                answer_text = str(response.text).strip()
                logger.info("✅ 응답 추출: response.text 사용")
            elif isinstance(response, str) and response.strip():
                answer_text = response.strip()
                logger.info("✅ 응답 추출: 직접 문자열 사용")
            # Gemini 특화 응답 처리
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    answer_text = response.candidates[0].content.parts[0].text.strip()
                    logger.info("✅ 응답 추출: Gemini candidates 사용")
                except Exception as e:
                    logger.error(f"❌ Gemini candidates 처리 실패: {e}")
            else:
                # 최후 수단: str() 변환하되 토큰화 방지
                raw_response = str(response)
                logger.warning(f"⚠️ 예상치 못한 응답 형식: {type(response)}")
                logger.info(f"🔍 Raw response preview: {raw_response[:200]}...")
                
                # 토큰화된 응답인지 확인 (공백으로 분리된 토큰들)
                if " " in raw_response and len(raw_response.split()) > 100:
                    logger.error("❌ 토큰화된 응답 감지됨 - 원본 텍스트 복원 시도")
                    # 토큰을 합쳐서 원본 텍스트로 복원 시도
                    answer_text = raw_response.replace(" ", "")
                else:
                    answer_text = raw_response
            
            # 최종 검증
            if not answer_text or len(answer_text.strip()) == 0:
                answer_text = f"죄송합니다. '{query}'에 대한 응답을 생성할 수 없습니다."
                logger.error("❌ 빈 응답 - 기본 메시지로 대체")
            
            logger.info(f"🎯 최종 응답 길이: {len(answer_text)} 문자")
            logger.info(f"🎯 응답 미리보기: {answer_text[:100]}...")
            
            state["answer"] = answer_text
            
            # 대화 기록 저장
            try:
                user_id = state["user_id"]
                logger.info(f"💾 대화 기록 저장 시작 - 사용자: {user_id}")
                logger.info(f"💾 저장할 질문 길이: {len(query)} 문자")
                logger.info(f"💾 저장할 답변 길이: {len(answer_text)} 문자")
                
                self.rag_system.chat_history_manager.add_to_chat_history(
                    user_id, query, answer_text
                )
                
                logger.info("✅ 대화 기록 저장 완료")
            except Exception as save_error:
                logger.error(f"❌ 대화 기록 저장 실패: {save_error}", exc_info=True)
            
            logger.info("🎯 답변 생성 완료")
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            state["error"] = f"답변 생성 실패: {str(e)}"
        
        return state
    
    @traceable(name="Final Prompt Construction", run_type="tool")
    async def _log_final_prompt(self, prompt: str, intent: str, context_info: Dict) -> Dict:
        """최종 프롬프트 정보를 LangSmith에 기록하고 콘솔에 미리보기를 출력합니다."""

        system_prompt = context_info.get("system_prompt") or ""
        task_prompt = context_info.get("task_prompt") or ""
        retrieved_documents = context_info.get("retrieved_documents") or ""
        conversation_history = context_info.get("conversation_history") or ""
        detected_language = context_info.get("detected_language", "unknown")

        # 프롬프트 미리보기 로그
        logger.info(f"📝 답변 생성 프롬프트 구성 (Intent: {intent}, Language: {detected_language})")
        logger.info(f"  총 길이: {len(prompt)} 문자")

        # 섹션별 길이 정보
        logger.info(f"  섹션별 구성:")
        logger.info(f"    - System Prompt: {len(system_prompt)} 문자")
        logger.info(f"    - Task Prompt: {len(task_prompt)} 문자")
        logger.info(f"    - Retrieved Documents: {len(retrieved_documents)} 문자")
        logger.info(f"    - Conversation History: {len(conversation_history)} 문자")

        logger.info(f"  프롬프트 미리보기:\n{prompt[:200]}")

        prompt_analysis = {
            "intent": intent,
            "final_prompt": prompt,
            "prompt_sections": {
                "system_prompt": system_prompt,
                "task_prompt": task_prompt,
                "retrieved_documents": retrieved_documents,
                "conversation_history": conversation_history
            }
        }

        return prompt_analysis
