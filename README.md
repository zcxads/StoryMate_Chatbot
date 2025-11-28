# StoryMate Chatbot API

## 프로젝트 개요

**StoryMate Chatbot API**는 사용자가 업로드한 문서를 기반으로 개인화된 독서 경험을 제공하는 AI 챗봇 시스템입니다. 하이브리드 검색(벡터 + BM25)과 다중 AI 모델을 통해 정확하고 맥락에 맞는 답변을 제공하며, 사용자별로 분리된 벡터 데이터베이스를 통해 개인정보를 보호합니다.

### 역할
FastAPI 기반 백엔드 API 서버로, 멀티턴 RAG(Retrieval-Augmented Generation) 시스템을 통해 개인화된 독서 챗봇 서비스를 제공합니다. LangGraph 기반 에이전트 아키텍처로 의도 분석, 문서 검색, 답변 생성을 관리합니다.

### 기술 스택
- **Backend**: Python 3.11, FastAPI 1.0.0, Uvicorn
- **AI/ML**: LangChain, LangGraph, OpenAI (GPT-4o, Embeddings), Google Gemini (2.0/2.5)
- **Vector Database**: Qdrant (벡터 데이터베이스)
- **Search**: BM25 (키워드 검색), Ensemble Retriever (하이브리드 검색)
- **Infrastructure**: Docker

## 주요 기능

- **멀티턴 RAG 시스템**: LangGraph 기반 에이전트 아키텍처로 대화 맥락을 유지하며 정확한 답변 생성
- **다중 AI 모델 지원**: OpenAI GPT-4o, Google Gemini 등 5개 모델 지원
- **하이브리드 검색**: 벡터(70%) + BM25(30%) 앙상블 검색으로 의미적/키워드 검색 결합
- **벡터 기반 메모리**: 모든 대화를 벡터로 임베딩하여 연관 대화 검색 및 맥락 유지
- **사용자별 데이터 격리**: `{user_id}_documents`, `{user_id}_chat` 컬렉션으로 완전 분리
- **문서 처리**: JSON 페이지 기반 업로드, 자동 청킹, 배치 처리 최적화
- **성능 최적화**: 임베딩/모델/검색기 캐싱, 비동기 처리

---

## 아키텍처 구조

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                        │
│                         (Mobile App)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                        │
│                   (app/main.py, port 8007)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Middleware: CORS, Logging                           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Router Layer                       │
│                    (app/api/v1/chat.py)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │   POST /chat/chat          # 챗봇 대화                │   │
│  │   POST /chat/documents/upload # 문서 업로드            │   │
│  │   GET  /chat/models        # 지원 모델 조회            │   │
│  │   POST /chat/documents/list # 문서 목록 조회           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   RAG System Layer                          │
│                 (app/core/system/rag_system.py)             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LangGraph Workflow (멀티턴 대화 관리)                 │   │
│  │                                                      │   │
│  │  ┌─────────────┐   ┌─────────────┐    ┌───────────┐  │   │
│  │  │ Intent      │──▶│ Context     │──▶│ Document  │  |   │
│  │  │ Analysis    │   │ Management  │    │ Retrieval │  │   │
│  │  │ Agent       │   │ Agent       │    │ Agent     │  │   │
│  │  └─────────────┘   └─────────────┘    └─────┬─────┘  │   │
│  │                                              │       │   │
│  │                    ┌─────────────────────────▼─────┐ │   │
│  │                    │ Answer Generation Agent      │  │   │
│  │                    └──────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Service Layer                      │
│  ┌───────────────┬──────────────┬──────────────────────┐    │
│  │ Chat Module   │ Document     │ LLM Module           │    │
│  │               │ Module       │                      │    │
│  │ - History Mgr │ - Loader     │ - Provider           │    │
│  │ - Intent      │ - Context    │ - Prompt Manager     │    │
│  │   Analyzer    │   Manager    │ - Answer Generator   │    │
│  └───────┬───────┴──────┬───────┴──────────┬───────────┘    │
│          │              │                  │                │
│  ┌───────▼──────────────▼──────────────────▼───────────┐    │
│  │           Retrieval Module                          │    │
│  │  - Hybrid Search (Vector + BM25)                    │    │
│  │  - Vector Store Manager (Qdrant)                    │    │
│  │  - Document Retriever Agent                         │    │
│  └─────────────────────┬───────────────────────────────┘    │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Services                         │
│  ┌────────────┬──────────────┬────────────────────────┐     │
│  │ OpenAI API             │  Google Gemini API       │      │
│  │ - GPT-4o               │  - Gemini 2.5 Pro        │      │
│  │ - GPT-4o-mini          │  - Gemini 2.5 Flash      │      │
│  └────────────────────────┴──────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Vector Database                         │
│                        (Qdrant)                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  User Collections (per user_id):                     │   │
│  │  - {user_id}_documents  # 문서 내용                   │   │
│  │  - {user_id}_chat       # 대화 기록                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 레이어별 설명

1. **Client Layer**: 클라이언트(모바일)에서 API 호출
2. **FastAPI Gateway**: 요청 라우팅, CORS, 로깅
3. **API Router Layer**: 채팅, 문서 업로드, 모델 조회 등 엔드포인트 정의
4. **RAG System Layer**: LangGraph 기반 멀티턴 대화 워크플로우 관리
5. **Core Service Layer**: 의도 분석, 문서 관리, LLM 제공, 하이브리드 검색
6. **External Services**: AI 모델 API (OpenAI, Gemini)
7. **Vector Database**: Qdrant - 사용자별 문서/채팅 컬렉션 관리

---

## Docker 배포 가이드라인

### 배포 단계

#### 1. 환경 설정
```bash
# .env.example을 복사하여 .env 파일 생성
cp .env.example .env

# .env 파일 편집 - 아래 API 키 설정
# OPENAI_API_KEY=your_openai_api_key_here
# GEMINI_API_KEY=your_gemini_api_key_here
```

#### 2. Docker 이미지 빌드 및 실행
```bash
# 컨테이너 빌드 및 시작
docker compose up -d --build

# 로그 확인
docker compose logs -f

# 서비스 상태 확인
docker compose ps
```

#### 3. 서비스 확인
```bash
# Health Check
curl http://localhost:8007/

# API 문서 확인
# 브라우저에서 http://localhost:8007/docs 접속
```

#### 4. 컨테이너 관리
```bash
# 컨테이너 중지
docker compose down

# 컨테이너 재시작
docker compose restart

# 로그 확인
docker compose logs -f chatbot-api

# 컨테이너 내부 접속 (디버깅)
docker compose exec chatbot-api bash
```

---

## 주요 디렉토리 구조

```
C:\StoryMate\chatbot-api/
│
├── app/                          # 메인 애플리케이션 디렉토리
│   ├── main.py                   # FastAPI 애플리케이션 엔트리포인트
│   ├── config.py                 # 애플리케이션 설정 및 환경 변수
│   │
│   ├── api/                      # API 라우터 레이어
│   │   └── v1/                   # API v1 엔드포인트
│   │       └── chat.py           # 채팅 및 문서 업로드 API
│   │
│   ├── core/                     # 핵심 비즈니스 로직
│   │   ├── chat/                 # 채팅 관련 모듈
│   │   │   ├── chat_history_manager.py  # 벡터 기반 대화 히스토리 관리
│   │   │   ├── intent_analyzer.py       # 질문 의도 분석
│   │   │   └── intent_analyzer_agent.py # 의도 분석 에이전트
│   │   │
│   │   ├── document/             # 문서 처리 모듈
│   │   │   ├── document_loader.py       # 문서 로딩 및 청킹
│   │   │   ├── document_context.py      # 문서 컨텍스트 관리
│   │   │   └── context_manager_agent.py # 컨텍스트 관리 에이전트
│   │   │
│   │   ├── llm/                  # LLM 관련 모듈
│   │   │   ├── llm_provider.py           # 다중 AI 모델 제공자
│   │   │   ├── prompt_manager.py         # 프롬프트 템플릿 관리
│   │   │   ├── answer_generator_agent.py # 답변 생성 에이전트
│   │   │   ├── hierarchical_intent_analyzer.py # 계층적 의도 분석
│   │   │   ├── fewshot_intent_classifier.py    # Few-shot 의도 분류
│   │   │   └── prompts/          # 프롬프트 템플릿 파일
│   │   │       ├── unified_rag.yaml                   # 통합 RAG 프롬프트
│   │   │       ├── multiturn_answer_generation.yaml   # 멀티턴 답변 생성
│   │   │       ├── multiturn_intent_analysis.yaml     # 멀티턴 의도 분석
│   │   │       ├── document_list.yaml                 # 문서 목록 조회
│   │   │       └── fewshot_intent_classification.yaml # Few-shot 의도 분류
│   │   │
│   │   ├── retrieval/            # 검색 관련 모듈
│   │   │   ├── retriever_manager.py       # 하이브리드 검색 관리
│   │   │   ├── vector_store.py            # Qdrant 벡터 데이터베이스 관리
│   │   │   └── document_retriever_agent.py # 문서 검색 에이전트
│   │   │
│   │   └── system/               # 시스템 통합 모듈
│   │       └── rag_system.py     # 멀티턴 RAG 시스템 (LangGraph 기반)
│   │
│   ├── models/                   # Pydantic 데이터 모델
│   │   └── chat.py               # 채팅 요청/응답 모델
│   │
│   ├── states/                   # LangGraph 상태 정의
│   │   └── state.py              # RAG 시스템 상태 모델
│   │
│   ├── utils/                    # 유틸리티 함수
│   │   ├── language_detector.py  # 언어 감지
│   │   └── model.py              # 모델 유틸리티
│   │
│   └── logs/                     # 로깅 설정
│       └── logger.py             # 로거 설정
│
├── .env.example                 # 환경 변수 예시
├── .gitignore                   # Git 무시 파일
├── .dockerignore                # Docker 무시 파일
├── Dockerfile                   # Docker 이미지 빌드 설정
├── docker-compose.yml           # Docker Compose 설정
├── requirements.txt             # Python 의존성
├── README.md                    # 프로젝트 문서
└── DEPLOYMENT.md               # 배포 가이드
```

---

## 로컬 개발 환경 설정

Docker 없이 로컬에서 직접 실행하는 방법입니다.

### 1. 사전 요구사항
- Python 3.11 이상
- pip 또는 poetry
- Qdrant 서버 (로컬 또는 클라우드)

### 2. 의존성 설치

```bash
# Python 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
# .env 파일 생성 및 편집
# 위의 "Docker 배포 가이드라인 > 환경 설정" 참고
```

### 4. Qdrant 서버 실행

```bash
# Docker로 Qdrant 실행 (로컬 개발용)
docker run -p 6333:6333 qdrant/qdrant

# 또는 클라우드 Qdrant 사용 (QDRANT_URL 환경 변수 설정)
```

### 5. 서버 실행

```bash
# 메인 애플리케이션 실행
cd app
python main.py

# 또는 uvicorn으로 직접 실행
uvicorn app.main:app --host 0.0.0.0 --port 8007 --reload
```

### 6. 서비스 확인

```bash
# API 상태 확인
curl http://localhost:8007/

# API 문서
# 브라우저에서 http://localhost:8007/docs 접속
```

---

## API 서비스 구조

### 1. 지원 모델 조회 - GET `/api/v1/chat/models`
- 지원하는 AI 모델 목록을 조회합니다.

---

### 2. 문서 업로드 - POST `/api/v1/chat/documents/upload`
- 사용자별로 문서를 업로드하고 벡터 데이터베이스에 저장합니다.

---

### 3. 챗봇 대화 - POST `/api/v1/chat/chat`
- 사용자 질문에 대해 문서 기반 답변을 생성합니다. 멀티턴 RAG 시스템으로 대화 맥락을 유지합니다.

---

### 4. 문서 목록 조회 - POST `/api/v1/chat/documents/list`
- 사용자가 업로드한 문서 목록을 조회합니다.

---

## 고급 기능

### 멀티턴 RAG 시스템 (LangGraph 기반)

시스템은 LangGraph를 사용한 에이전트 기반 아키텍처로 멀티턴 대화를 관리합니다:

- **에이전트 기반 아키텍처**: 의도 분석, 컨텍스트 관리, 문서 검색, 답변 생성을 각각 에이전트화
- **LangGraph 워크플로우**: 상태 기반 그래프로 멀티턴 대화 관리
- **향상된 의도 분석**: 대화 히스토리를 고려한 정확한 사용자 의도 파악
- **컨텍스트 초기화**: 새로운 대화 시작 시 자동으로 이전 맥락 리셋

### 하이브리드 검색 시스템

시스템은 벡터 검색과 BM25 검색을 결합한 앙상블 검색을 사용합니다:

- **벡터 검색 (70%)**: OpenAI Embeddings를 사용한 의미적 유사성 기반 검색
- **BM25 검색 (30%)**: 키워드 기반 정확 매칭
- **자동 샘플링**: BM25용 문서 1000개 샘플링으로 성능 최적화
- **동적 파라미터 조정**: 쿼리 길이와 컬렉션 크기에 따른 검색 범위 자동 조정
- **캐싱**: 검색기 재사용으로 응답 속도 향상

### 벡터 기반 메모리 시스템

- **자동 메모리 저장**: 모든 대화 내용을 벡터로 임베딩하여 채팅 전용 벡터 DB에 저장
- **연관 대화 검색**: 새로운 발화와 의미적으로 연관된 이전 대화를 벡터 유사도로 검색
- **맥락 유지**: 연관된 이전 대화를 프롬프트에 추가하여 일관성 있는 응답 생성
- **메모리 최적화**: 최근 20개 대화 유지, 오래된 대화 자동 제거
- **대화-문서 통합**: 벡터 기반 메모리와 문서 내용을 조합한 하이브리드 검색

### 사용자별 데이터 격리

시스템은 문서와 채팅 내용을 사용자별로 완전히 분리된 컬렉션에서 관리합니다:

- **문서 컬렉션**: `{user_id}_documents`
  - 업로드된 문서 내용만 저장
  - 도서별로 구조화된 정보 관리 (book_id, page_key)
  - 문서 검색 및 RAG에 사용

- **채팅 컬렉션**: `{user_id}_chat`
  - 사용자와 AI의 대화 내용만 저장
  - 벡터 기반 메모리 검색에 사용
  - 이전 대화 내용 참조를 위한 연관성 검색

**장점:**
- 데이터 격리로 보안성 향상
- 검색 성능 최적화 (문서와 채팅 분리 검색)
- 관리 용이성 (각각 독립적인 컬렉션)
- 확장성 (문서와 채팅 각각 독립적 확장)

### 성능 최적화

- **임베딩 캐싱**: LRU 캐싱으로 중복 임베딩 방지
- **모델 캐싱**: AI 모델 인스턴스 재사용
- **검색기 캐싱**: 벡터 검색기 및 BM25 검색기 재사용
- **배치 처리**: 대용량 문서 처리 최적화
- **비동기 처리**: FastAPI 기반 고성능 비동기 처리

---

## 문제 해결

### 자주 발생하는 문제

1. **Qdrant 연결 실패**
   ```
   해결: Qdrant 서버 상태 확인, QDRANT_URL 환경 변수 확인
   curl http://localhost:6333/
   ```

2. **API 키 오류**
   ```
   해결: .env 파일에 올바른 API 키 설정
   OPENAI_API_KEY, GEMINI_API_KEY 확인
   ```

3. **메모리 부족**
   ```
   해결: 청크 크기 조정
   CHUNK_SIZE=500
   CHUNK_OVERLAP=100
   ```

4. **검색 성능 저하**
   ```
   해결: 검색 파라미터 조정
   DEFAULT_SEARCH_K=20
   DEFAULT_SCORE_THRESHOLD=0.6
   ```

5. **포트 충돌**
   ```
   해결: docker-compose.yml 또는 .env에서 포트 변경
   SERVER_PORT=8008
   ```

---

## API 문서

- **Swagger UI**: http://localhost:8007/docs
- **Root Endpoint**: http://localhost:8007/
