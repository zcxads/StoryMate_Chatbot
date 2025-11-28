import os
from dotenv import load_dotenv

from app.logs.logger import setup_logger

logger = setup_logger('config')

# .env 파일에서 환경 변수 로드
load_dotenv()

class Settings:
    """애플리케이션 설정을 관리하는 클래스"""
    # 환경 설정
    DEBUG: bool = os.getenv("DEBUG", "true") == "true"  # 디버그 모드
    
    # API 키 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Qdrant 벡터 데이터베이스 설정
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_URL")

    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME")
    
    # 사용자별 벡터 DB 설정
    USER_COLLECTION_SUFFIX: str = os.getenv("USER_COLLECTION_SUFFIX", "_documents")
    
    # Qdrant 타임아웃 설정
    QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT"))

    # 문서 처리 설정
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP"))
    
    # 검색 성능 최적화 설정
    MAX_SEARCH_TIME: int = int(os.getenv("MAX_SEARCH_TIME"))
    
    # ChatGenerator 설정
    DEFAULT_SEARCH_K: int = int(os.getenv("DEFAULT_SEARCH_K", "50"))  # 기본 검색 문서 수
    DEFAULT_SCORE_THRESHOLD: float = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.8"))  # 기본 점수 임계값
    DEFAULT_SEARCH_TIMEOUT: int = int(os.getenv("DEFAULT_SEARCH_TIMEOUT", "10"))  # 기본 검색 타임아웃
    MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "20"))  # 최대 대화 히스토리 수
    ENSEMBLE_VECTOR_WEIGHT: float = float(os.getenv("ENSEMBLE_VECTOR_WEIGHT", "0.7"))  # 앙상블 검색 벡터 가중치
    ENSEMBLE_BM25_WEIGHT: float = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.3"))  # 앙상블 검색 BM25 가중치
    BM25_SAMPLE_SIZE: int = int(os.getenv("BM25_SAMPLE_SIZE", "1000"))  # BM25 샘플링 크기
    BM25_K_RATIO: float = float(os.getenv("BM25_K_RATIO", "0.3"))  # BM25 k 비율

    # Model 설정
    MODEL_MAX_TOKENS: int = int(os.getenv("MODEL_MAX_TOKENS", "2000"))  # 모델 최대 토큰 수
    MODEL_REQUEST_TIMEOUT: int = int(os.getenv("MODEL_REQUEST_TIMEOUT", "10"))  # 모델 요청 타임아웃
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 임베딩 모델

    # 의도 분석용 모델 설정
    INTENT_ANALYSIS_MODEL: str = os.getenv("INTENT_ANALYSIS_MODEL", "gpt-4o")  # 의도 분석용 메인 모델
    INTENT_ANALYSIS_FALLBACK_MODEL: str = os.getenv("INTENT_ANALYSIS_FALLBACK_MODEL", "gpt-4o-mini")  # 의도 분석용 폴백 모델

    # 답변 생성용 모델 설정
    ANSWER_GENERATION_MODEL: str = os.getenv("ANSWER_GENERATION_MODEL", "gpt-4o")  # 답변 생성용 기본 모델

    # 채팅 API 기본 모델
    DEFAULT_CHAT_MODEL: str = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o")  # 채팅 API 기본 모델

    # 서버 설정
    SERVER_HOST: str = os.getenv("SERVER_HOST")  # 서버 호스트 IP
    SERVER_PORT: int = int(os.getenv("SERVER_PORT"))  # 서버 포트

    def __init__(self) -> None:
        """설정 초기화"""
        logger.info(f"디버그 모드: {self.DEBUG}")
        logger.info(f"Qdrant URL (고정): {self.QDRANT_URL}")
        logger.info(f"Qdrant Collection (기본): {self.QDRANT_COLLECTION_NAME}")
        logger.info(f"사용자 컬렉션 접미사: {self.USER_COLLECTION_SUFFIX}")

    def get_user_collection_name(self, user_id: str) -> str:
        """사용자별 동적 컬렉션명 생성"""
        return f"{user_id}{self.USER_COLLECTION_SUFFIX}"

# 전역 설정 인스턴스 생성
settings = Settings()
