from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import re
import uuid
from app.config import settings

# 지원되는 모델 목록
SUPPORTED_CHAT_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4o",
    "gpt-4o-mini"
]

# 지원되는 캐릭터 장르 목록 (총 22가지)
SUPPORTED_CHARACTER_GENRES = [
    # 알 상태 (0레벨)
    "EGG",                      # 알 상태 - 기본 말투

    # 순혈 장르 (6종)
    "SCIENCE",                  # 과학 - 닥터 로직스 계열
    "HISTORY",                  # 역사 - 크로노스 계열
    "PHILOSOPHY",               # 철학 - 소피아 계열
    "LITERATURE",               # 문학 - 뮤즈 계열
    "ART",                      # 예술 - 아르테 계열
    "FICTION",                  # 허구 - 판타지아 계열

    # 혼혈 장르 (15종)
    "SCIENCE_HISTORY",          # 과학+역사 - 아카데미쿠스 계열
    "SCIENCE_PHILOSOPHY",       # 과학+철학 - 사이언티아 계열
    "SCIENCE_LITERATURE",       # 과학+문학 - 포에티카 계열
    "SCIENCE_ART",              # 과학+예술 - 테크니카 계열
    "SCIENCE_FICTION",          # 과학+허구 - 이매지나리오 계열
    "HISTORY_PHILOSOPHY",       # 역사+철학 - 템포랄리스 계열
    "HISTORY_LITERATURE",       # 역사+문학 - 나라티바 계열
    "HISTORY_ART",              # 역사+예술 - 헤리티지 계열
    "HISTORY_FICTION",          # 역사+허구 - 타임워커 계열
    "PHILOSOPHY_LITERATURE",    # 철학+문학 - 로고스 계열
    "PHILOSOPHY_ART",           # 철학+예술 - 아에스테시스 계열
    "PHILOSOPHY_FICTION",       # 철학+허구 - 메타피지카 계열
    "LITERATURE_ART",           # 문학+예술 - 크리에이토 계열
    "LITERATURE_FICTION",       # 문학+허구 - 나라티바 계열 (2)
    "ART_FICTION"               # 예술+허구 - 비주얼마법사 계열
]

class TextInput(BaseModel):
    pageKey: int
    text: str
    
    @validator('text')
    def clean_text(cls, v):
        """텍스트에서 제어 문자를 정제하고 정규화합니다."""
        if v is None:
            return ""
        
        # 제어 문자 제거 (줄바꿈, 탭 등은 유지)
        # 유효한 제어 문자만 허용: \n, \r, \t
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', str(v))
        
        # 연속된 공백 정규화
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # 연속된 줄바꿈 정규화
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()

class ChunkInfo(BaseModel):
    """개별 청크 정보를 담는 모델"""
    pageKey: int
    text: str

class DocumentUploadRequest(BaseModel):
    user_id: str
    book_id: int
    pages: List[TextInput]

class DocumentUploadResponse(BaseModel):
    user_id: str
    message: str

class ChatRequest(BaseModel):
    user_id: str
    query: str
    session_id: Optional[str] = Field(default="")
    model: Optional[str] = Field(default=settings.DEFAULT_CHAT_MODEL)
    character_genre: Optional[str] = Field(default=None, description="캐릭터 장르 (SCIENCE, HISTORY, PHILOSOPHY, LITERATURE, ART, FICTION)")

    @validator('model')
    def validate_model(cls, v):
        if v not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"지원되지 않는 모델입니다. 지원 모델: {', '.join(SUPPORTED_CHAT_MODELS)}")
        return v

    @validator('character_genre')
    def validate_character_genre(cls, v):
        if v is not None and v not in SUPPORTED_CHARACTER_GENRES:
            raise ValueError(f"지원되지 않는 장르입니다. 지원 장르: {', '.join(SUPPORTED_CHARACTER_GENRES)}")
        return v

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 특정 사용자 문서 상세 정보 조회를 위한 모델
class UserDocumentDetailResponse(BaseModel):
    user_id: str
    books: List[Dict[str, Any]]  # 여러 도서 정보를 담을 수 있도록 변경
    total_books: int
    message: str
