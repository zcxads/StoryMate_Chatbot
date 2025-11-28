from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

from app.api.v1 import chat
from app.logs.logger import setup_logger
from app.config import settings

logger = setup_logger('main')

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="StoryMate 챗봇 API",
    description="개인화 독서 챗봇 API - 문서 업로드, 벡터 DB 기반 검색 및 답변 생성",
    version="1.0.0",
    docs_url="/docs",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "StoryMate 챗봇 API에 오신 것을 환영합니다",
        "version": "1.0.0",
        "docs_url": "/docs",
        "features": [
            "문서 업로드 및 벡터 DB 저장",
            "벡터 기반 의미적 검색",
            "사용자별 개인화된 답변 생성"
        ]
    }

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 HTTP 요청과 응답을 로깅하는 미들웨어"""
    # 요청 로깅
    logger.info(f"요청: {request.method} {request.url}")
    
    # 응답 처리
    response = await call_next(request)
    
    # 응답 로깅
    logger.info(f"응답: {response.status_code}")
    
    return response

def run_server():
    """서버 실행 함수"""
    # 현재 디렉토리를 Python 경로에 추가
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # 서버 실행 설정
    host = settings.SERVER_HOST
    port = settings.SERVER_PORT
    
    logger.info(f"🚀 StoryMate 챗봇 서버 시작 중...")
    logger.info(f"📍 서버 주소: http://{host}:{port}")
    logger.info(f"📚 API 문서: http://{host}:{port}/docs")
    logger.info(f"🌐 네트워크 접근 가능: 다른 기기에서 {host}:{port}로 접속 가능")
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=settings.DEBUG,  # 환경에 따라 reload 기능 제어
        log_level="warning",  # uvicorn 로그 레벨을 warning으로 설정하여 중복 방지
        access_log=False  # access log 비활성화하여 중복 방지
    )

if __name__ == "__main__":
    run_server()
