from typing import Optional, List, Any, Dict, Union
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import anthropic
import google.generativeai as genai

from app.logs.logger import setup_logger
from app.config import settings

logger = setup_logger('model')


# 상수 정의 (config에서 가져옴)
@dataclass
class ModelConfig:
    """모델 설정을 위한 데이터 클래스"""
    MAX_TOKENS: int = settings.MODEL_MAX_TOKENS
    REQUEST_TIMEOUT: int = settings.MODEL_REQUEST_TIMEOUT


# 모델 이름 매핑
GEMINI_MODEL_MAPPING = {
    "gemini-2.5-pro": "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash": "gemini-2.5-flash-preview-04-17",
    "gemini-2.0-flash": "gemini-2.0-flash-exp"
}

CLAUDE_MODEL_MAPPING = {
    "claude-opus-4-0": "claude-opus-4-20250514",
    "claude-sonnet-4-0": "claude-sonnet-4-20250514"
}

SUPPORTED_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "claude": ["claude-sonnet-4-0", "claude-opus-4-0"]
}


class ModelError(Exception):
    """모델 관련 예외 클래스"""
    pass


class GeminiChatModel(BaseChatModel):
    """Google Gemini 모델을 위한 LangChain 호환 래퍼"""
    
    def __init__(self, model_name: str, temperature: float = settings.TEMPERATURE, api_key: str = settings.GEMINI_API_KEY):
        super().__init__()
        
        if not api_key:
            raise ModelError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # __dict__를 사용하여 필드 설정 (Pydantic 제약 우회)
        self.__dict__['model_name'] = model_name
        self.__dict__['temperature'] = temperature
        self.__dict__['api_key'] = api_key
        
        # Gemini 설정
        genai.configure(api_key=self.api_key)
        self.__dict__['model'] = genai.GenerativeModel(model_name)
    
    def _convert_messages_to_prompt(self, messages: List[Any]) -> str:
        """LangChain 메시지를 Gemini 프롬프트로 변환합니다."""
        prompt_parts = []
        
        for message in messages:
            if hasattr(message, 'content'):
                if isinstance(message, SystemMessage):
                    prompt_parts.append(f"System: {message.content}")
                elif isinstance(message, HumanMessage):
                    prompt_parts.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    prompt_parts.append(f"Assistant: {message.content}")
                else:
                    prompt_parts.append(str(message.content))
            else:
                prompt_parts.append(str(message))
        
        return "\n".join(prompt_parts)
    
    def _generate_response(self, messages: List[Any]) -> str:
        """Gemini 모델을 사용하여 응답을 생성합니다."""
        try:
            prompt = self._convert_messages_to_prompt(messages)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=ModelConfig.MAX_TOKENS
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini 모델 호출 실패: {e}")
            raise ModelError(f"Gemini 모델 호출 중 오류 발생: {e}")
    
    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> ChatResult:
        """LangChain 호환 메시지 생성 메서드"""
        try:
            response_text = self._generate_response(messages)
            
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=response_text)
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Gemini 메시지 생성 실패: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def invoke(self, input_data, config: Optional[Dict[str, Any]] = None, **kwargs) -> AIMessage:
        """LangChain 0.2.x 호환 invoke 메서드 - 문자열과 메시지 리스트 모두 지원"""
        try:
            # 문자열 입력 처리
            if isinstance(input_data, str):
                logger.info("🔍 문자열 프롬프트 직접 처리")
                response = self.model.generate_content(
                    input_data,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=ModelConfig.MAX_TOKENS
                    )
                )
                
                # 응답 텍스트 안전하게 추출
                response_text = ""
                if hasattr(response, 'text') and response.text:
                    response_text = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    try:
                        response_text = response.candidates[0].content.parts[0].text.strip()
                    except Exception as e:
                        logger.error(f"❌ Gemini candidates 추출 실패: {e}")
                        response_text = "응답 생성 중 오류가 발생했습니다."
                
                return AIMessage(content=response_text)
            
            # 메시지 리스트 입력 처리 (기존 방식)
            else:
                logger.info("🔍 메시지 리스트 처리")
                result = self._generate(input_data, **kwargs)
                return result.generations[0].message
                
        except Exception as e:
            logger.error(f"❌ Gemini invoke 실패: {e}")
            return AIMessage(content=f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}")


class ClaudeChatModel(BaseChatModel):
    """Anthropic Claude 모델을 위한 LangChain 호환 래퍼"""
    
    def __init__(self, model_name: str, temperature: float = settings.TEMPERATURE, api_key: str = settings.ANTHROPIC_API_KEY):
        super().__init__()
        
        if not api_key:
            raise ModelError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # __dict__를 사용하여 필드 설정 (Pydantic 제약 우회)
        self.__dict__['model_name'] = model_name
        self.__dict__['temperature'] = temperature
        self.__dict__['api_key'] = api_key
        
        # Claude 클라이언트 초기화
        self.__dict__['client'] = anthropic.Client(api_key=self.api_key)
    
    def _convert_messages_to_claude_format(self, messages: List[Any]) -> List[Dict[str, str]]:
        """LangChain 메시지를 Claude 형식으로 변환합니다."""
        claude_messages = []
        
        for message in messages:
            if hasattr(message, 'content'):
                if isinstance(message, SystemMessage):
                    claude_messages.append({"role": "user", "content": f"System: {message.content}"})
                elif isinstance(message, HumanMessage):
                    claude_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    claude_messages.append({"role": "assistant", "content": message.content})
                else:
                    claude_messages.append({"role": "user", "content": str(message.content)})
            else:
                claude_messages.append({"role": "user", "content": str(message)})
        
        return claude_messages
    
    def _generate_response(self, messages: List[Any]) -> str:
        """Claude 모델을 사용하여 응답을 생성합니다."""
        try:
            claude_messages = self._convert_messages_to_claude_format(messages)
            
            response = self.client.messages.create(
                model=self.model_name,
                messages=claude_messages,
                max_tokens=ModelConfig.MAX_TOKENS,
                temperature=self.temperature
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude 모델 호출 실패: {e}")
            raise ModelError(f"Claude 모델 호출 중 오류 발생: {e}")
    
    def _generate(self, messages: List[Any], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs) -> ChatResult:
        """LangChain 호환 메시지 생성 메서드"""
        try:
            response_text = self._generate_response(messages)
            
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=response_text)
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Claude 메시지 생성 실패: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "claude"
    
    def invoke(self, messages: List[Any], config: Optional[Dict[str, Any]] = None, **kwargs) -> AIMessage:
        """LangChain 0.2.x 호환 invoke 메서드"""
        result = self._generate(messages, **kwargs)
        return result.generations[0].message

class ModelFactory:
    """다양한 AI 모델을 생성하는 팩토리 클래스"""
    
    @staticmethod
    def get_supported_models() -> Dict[str, List[str]]:
        """지원되는 모델 목록을 반환합니다."""
        return SUPPORTED_MODELS.copy()
    
    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """모델이 지원되는지 확인합니다."""
        for models in SUPPORTED_MODELS.values():
            if model_name in models:
                return True
        return False
    
    @staticmethod
    def create_llm(model_name: str) -> Optional[Union[ChatOpenAI, GeminiChatModel, ClaudeChatModel]]:
        """
        모델 이름에 따라 적절한 LLM 인스턴스를 생성합니다.
        
        Args:
            model_name: 모델 이름 (예: "gpt-4o", "gemini-2.0-flash", "claude-sonnet-4-0")
            
        Returns:
            LLM 인스턴스 또는 None (지원되지 않는 모델인 경우)
            
        Raises:
            ModelError: 모델 생성 중 오류가 발생한 경우
        """
        try:
            if not ModelFactory.is_model_supported(model_name):
                logger.error(f"지원되지 않는 모델: {model_name}")
                return None
            
            if model_name.startswith("gpt-"):
                return ModelFactory._create_openai_model(model_name)
            elif model_name.startswith("gemini-"):
                return ModelFactory._create_gemini_model(model_name)
            elif model_name.startswith("claude-"):
                return ModelFactory._create_claude_model(model_name)
            else:
                logger.error(f"알 수 없는 모델 형식: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"모델 생성 실패 ({model_name}): {e}")
            raise ModelError(f"모델 생성 실패: {e}")
    
    @staticmethod
    def _create_openai_model(model_name: str) -> ChatOpenAI:
        """OpenAI 모델을 생성합니다."""
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise ModelError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        return ChatOpenAI(
            model=model_name,
            temperature=settings.TEMPERATURE,
            api_key=api_key,
            max_tokens=ModelConfig.MAX_TOKENS,
            request_timeout=ModelConfig.REQUEST_TIMEOUT
        )
    
    @staticmethod
    def _create_gemini_model(model_name: str) -> GeminiChatModel:
        """Google Gemini 모델을 생성합니다."""
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ModelError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 실제 모델 이름으로 매핑
        actual_model_name = GEMINI_MODEL_MAPPING.get(model_name, model_name)
        
        return GeminiChatModel(
            model_name=actual_model_name,
            temperature=settings.TEMPERATURE,
            api_key=api_key
        )
    
    @staticmethod
    def _create_claude_model(model_name: str) -> ClaudeChatModel:
        """Anthropic Claude 모델을 생성합니다."""
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ModelError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # 실제 모델 이름으로 매핑
        actual_model_name = CLAUDE_MODEL_MAPPING.get(model_name, model_name)
        
        return ClaudeChatModel(
            model_name=actual_model_name,
            temperature=settings.TEMPERATURE,
            api_key=api_key
        )
