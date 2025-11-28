from fastapi import APIRouter, HTTPException, status
import time
import json
import re
import uuid

from app.models.chat import (
    ChatRequest,
    ChatResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    UserDocumentDetailResponse,
    SUPPORTED_CHAT_MODELS,
    ChunkInfo
)
from app.core.system.rag_system import MultiturnRAGSystem
from app.logs.logger import setup_logger
from app.config import settings

router = APIRouter()
rag_system = MultiturnRAGSystem()

logger = setup_logger('chat')

@router.get("/models")
async def get_supported_models():
    """
    요약에서 지원되는 AI 모델 목록을 반환합니다.

    Returns:
        SupportedModelsResponse: 지원되는 모델 목록과 기본 모델 정보
    """
    return {
        "supported_models": SUPPORTED_CHAT_MODELS,
        "default_model": settings.DEFAULT_CHAT_MODEL,
        "total_count": len(SUPPORTED_CHAT_MODELS)
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """LangGraph 기반 채팅을 수행합니다."""
    api_start_time = time.time()

    # session_id가 비어있으면 8자리 UUID 생성
    session_id = request.session_id
    if not session_id or session_id.strip() == "":
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"새로운 session_id 생성: {session_id}")

    try:
        logger.info(f"💬 [세션 {session_id}] 멀티턴 RAG 채팅 요청 시작 - 사용자: {request.user_id}, 모델: {request.model}")
        
        # 동시성 모니터링: 현재 활성 요청 수 추적
        import threading
        if not hasattr(chat, '_active_requests'):
            chat._active_requests = 0
            chat._request_lock = threading.Lock()
        
        with chat._request_lock:
            chat._active_requests += 1
            current_requests = chat._active_requests
        
        logger.info(f"🔄 멀티턴 RAG 동시 요청 수: {current_requests} (사용자: {request.user_id})")

        # 사용자별 컬렉션을 사용한 답변 생성
        from app.states import MultiturnRAGState

        # 대화 히스토리 로드
        conversation_history = []
        try:
            # ChatHistoryManager에서 사용자의 대화 기록 가져오기
            user_chat_history = rag_system.chat_history_manager._chat_history.get(request.user_id, [])

            # 포맷 변환: {"query": "...", "response": "..."} → {"user": "...", "assistant": "..."}
            conversation_history = [
                {"user": conv.get("query", ""), "assistant": conv.get("response", "")}
                for conv in user_chat_history
            ]

            if conversation_history:
                logger.info(f"📚 [세션 {session_id}] 대화 기록 로드 완료: {len(conversation_history)}개 대화")
            else:
                logger.info(f"📚 [세션 {session_id}] 대화 기록 없음 (신규 사용자 또는 첫 대화)")

        except Exception as history_error:
            logger.error(f"❌ [세션 {session_id}] 대화 기록 로드 실패: {history_error}")
            conversation_history = []

        initial_state = MultiturnRAGState(
            user_id=request.user_id,
            query=request.query,
            conversation_history=conversation_history,
            intent=None,
            search_context=None,
            retrieved_documents=None,
            answer=None,
            character_genre=request.character_genre,  # 캐릭터 장르 추가
            reference_index=None,  # 대화 참조 인덱스
            reference_type=None    # 참조 유형
        )

        # 의도 분석
        intent_result = rag_system.intent_analyzer_agent.analyze_intent(initial_state)

        logger.info(f"📊 [세션 {session_id}] 의도 분석 결과: {intent_result.get('intent')}")
        if intent_result.get('reference_index') is not None:
            logger.info(f"📍 [세션 {session_id}] 대화 참조: index={intent_result.get('reference_index')}, type={intent_result.get('reference_type')}")

        # 의도에 따른 routing
        intent = intent_result.get("intent")

        if intent == "follow_up_summary":
            # follow_up_summary: handle_follow_up_request 호출
            logger.info(f"🔄 [세션 {session_id}] follow_up_summary 감지 → handle_follow_up_request 호출")
            retrieval_result = rag_system.handle_follow_up_request(intent_result)
            # retrieved_documents를 state에 병합
            intent_result.update(retrieval_result)
            final_result = await rag_system.answer_generator_agent.generate_answer(intent_result)

        elif intent == "general_chat":
            # general_chat: 문서 검색 없이 바로 답변 생성
            logger.info(f"💬 [세션 {session_id}] general_chat 감지 → 문서 검색 생략")
            final_result = await rag_system.answer_generator_agent.generate_answer(intent_result)

        else:
            # 그 외: 일반 문서 검색 → 답변 생성
            logger.info(f"📚 [세션 {session_id}] {intent} 감지 → 문서 검색 실행")
            retrieval_result = await rag_system.document_retriever_agent.retrieve_documents(intent_result)
            final_result = await rag_system.answer_generator_agent.generate_answer(retrieval_result)

        result = {"answer": final_result.get("answer", ""), "intent": final_result.get("intent")}
        
        execution_time = f"{time.time() - api_start_time:.2f}s"
        logger.info(f"✅ [세션 {session_id}] 멀티턴 RAG 채팅 응답 완료 - 실행시간: {execution_time}")
        
        # 의도 분석 결과 포함
        response_data = {
            "response": result.get("answer", "죄송합니다. 응답을 생성할 수 없습니다."),
            "session_id": session_id
        }
        
        # # 의도 분석 정보가 있으면 포함
        # if "intent_analysis" in result:
        #     response_data["intent_analysis"] = result["intent_analysis"]
        
        # return ChatResponse(**response_data)
            
    ### woony : return type 스트링 강제처리 및 불필요한 개행처리 제거 
        response_data["response"] = re.sub(r'([가-힣A-Za-z0-9.,!?\'"()])\n([가-힣A-Za-z0-9])', r'\1\2', response_data["response"])

    # 단일 개행은 공백으로 변환, 문단 구분(\n\n)은 유지
        response_data["response"] = re.sub( r'(?<!\n)\n(?!\n)',' ', response_data["response"])

        if "intent_analysis" in result:
            intent = result["intent_analysis"]
    
        # 문자열이 아니면 JSON 문자열로 변환
            if not isinstance(intent, str):
                intent = json.dumps(intent, ensure_ascii=False)
    
            response_data["intent_analysis"] = intent

        return ChatResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        execution_time = f"{time.time() - api_start_time:.2f}s"
        logger.error(f"멀티턴 RAG 채팅 처리 중 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"멀티턴 RAG 채팅 처리 중 오류가 발생했습니다: {str(e)}"
        )
    finally:
        # 요청 완료 시 카운터 감소
        with chat._request_lock:
            chat._active_requests = max(0, chat._active_requests - 1)
            logger.info(f"🔄 멀티턴 RAG 요청 완료 - 남은 동시 요청 수: {chat._active_requests}")

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: DocumentUploadRequest):
    """사용자 문서를 업로드하고 벡터 DB에 저장합니다."""
    start_time = time.time()
    
    try:
        logger.info(f"📚 문서 업로드 시작 - 사용자: {request.user_id}, 도서: {request.book_id}, 페이지 수: {len(request.pages)}")
        
        if not request.pages:
            raise HTTPException(status_code=400, detail="페이지 데이터가 없습니다.")
        
        # 페이지 데이터 검증
        total_text_length = 0
        valid_pages = 0
        for i, page in enumerate(request.pages):
            if hasattr(page, 'text'):
                text = page.text
            elif isinstance(page, dict):
                text = page.get('text', '')
            else:
                continue
                
            if text and text.strip():
                total_text_length += len(text)
                valid_pages += 1
            else:
                logger.warning(f"⚠️ 페이지 {i+1}의 텍스트가 비어있음")
        
        if valid_pages == 0:
            raise HTTPException(status_code=400, detail="유효한 페이지가 없습니다.")
        
        book_key = request.book_id
        pages = request.pages
                                 
        # 벡터DB 생성
        try:
            logger.info(f"📚 도서 {book_key} 문서 처리 시작")
            
            # 1단계: 페이지별로 개별 Document 생성
            documents = rag_system.load_documents_from_pages(
                pages=pages,
                user_id=request.user_id,
                book_id=book_key
            )
            
            if not documents:
                logger.warning(f"   ❌ 도서 {book_key}에서 처리할 문서가 없습니다.")
                raise HTTPException(status_code=400, detail="처리할 문서가 없습니다.")
                        
            # 2단계: 문서를 청크로 분할            
            chunks = rag_system.create_chunks(documents)
            
            if not chunks:
                logger.warning(f"   ❌ 도서 {book_key}에서 청크를 생성할 수 없습니다.")
                raise HTTPException(status_code=400, detail="문서에서 청크를 생성할 수 없습니다.")
            
            try:
                # 사용자별 벡터 DB에 청크 저장
                success = rag_system.create_vector_db(chunks, request.user_id, book_key)

                if success:
                    logger.info(f"      ✅ 벡터 DB 생성 완료!")
                else:
                    raise HTTPException(status_code=500, detail="벡터 DB 생성에 실패했습니다.")
                    
            except Exception as vector_error:
                raise HTTPException(status_code=500, detail=f"벡터 DB 생성 중 오류가 발생했습니다: {str(vector_error)}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"사용자 {request.user_id} 도서 {book_key} 문서 처리 실패: {e}")
            raise HTTPException(status_code=500, detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}")
        
        execution_time = time.time() - start_time
        logger.info(f"🎉 문서 업로드 완료!")
        logger.info(f"   📊 전체 처리 요약:")
        logger.info(f"      • 사용자: {request.user_id}")
        logger.info(f"      • 도서: {book_key}")
        logger.info(f"      • 원본 페이지 수: {len(pages)}개")
        logger.info(f"      • 유효 페이지 수: {valid_pages}개")
        logger.info(f"      • 총 텍스트 길이: {total_text_length:,} 문자")
        logger.info(f"      • 생성된 청크 수: {len(chunks) if 'chunks' in locals() else 'N/A'}")
        logger.info(f"      • 총 실행 시간: {execution_time:.2f}초")
        
        # 성공 메시지 정의
        message = f"사용자 {request.user_id}의 도서 {book_key} 문서 업로드가 완료되었습니다. ({valid_pages}개 페이지, {len(chunks) if 'chunks' in locals() else 0}개 청크)"
        
        return DocumentUploadResponse(
            user_id=request.user_id,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = f"{time.time() - start_time:.2f}s"
        logger.error(f"문서 업로드 중 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"문서 업로드 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/health")
async def chat_health_check():
    """채팅 서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "chat",
        "message": "채팅 서비스가 정상적으로 동작하고 있습니다."
    }

@router.get("/documents/user/{user_id}", response_model=UserDocumentDetailResponse)
async def get_user_document_detail(user_id: str):
    """특정 사용자의 문서 상세 정보를 조회합니다."""
    try:
        logger.info(f"사용자 {user_id} 문서 상세 정보 조회 시작")
        
        # 사용자의 문서 상세 정보 조회
        user_document_detail = rag_system.get_user_document_detail(user_id)
        
        if not user_document_detail:
            raise HTTPException(
                status_code=404,
                detail=f"사용자 {user_id}의 문서 정보를 찾을 수 없습니다."
            )
        
        # 응답 모델에 맞게 변환
        books = user_document_detail["books"]
        total_books = user_document_detail.get("total_books", 0)
        
        if not books:
            raise HTTPException(
                status_code=404,
                detail=f"사용자 {user_id}의 도서 정보를 찾을 수 없습니다."
            )
        
        # 모든 도서 정보를 처리
        response_books = []
        
        for book in books:
            book_id = int(book["bookKey"]) if book["bookKey"].isdigit() else 0
            chunks = book.get("chunks", [])
            
            # book_id 기준으로 ChunkInfo 모델로 변환
            chunk_infos = []
            if chunks and len(chunks) > 0:
                # chunks에서 청크 정보 추출
                for chunk in chunks:
                    page_key = chunk.get("pageKey", 1)
                    text = chunk.get("text", "")
                    
                    # text가 비어있는 경우 기본값 설정
                    if not text:
                        text = "문서 내용을 불러올 수 없습니다."
                        logger.warning(f"⚠️ book_id {book_id}, pageKey {page_key}: text가 비어있음")
                    
                    chunk_infos.append(ChunkInfo(
                        pageKey=page_key,
                        text=text
                    ))
                
            else:
                logger.warning(f"⚠️ book_id {book_id}: chunks가 비어있음, 기본 청크 생성")
                # chunks가 비어있는 경우 기본 청크 생성
                chunk_infos.append(ChunkInfo(
                    pageKey=1,
                    text="문서 내용을 불러올 수 없습니다.",
                ))
            
            # 도서 정보를 딕셔너리로 변환 (카테고리 정보 포함)
            book_info = {
                "book_id": book_id,
                "chunks": chunk_infos,
            }
            response_books.append(book_info)
        
        return UserDocumentDetailResponse(
            user_id=user_id,
            books=response_books,
            total_books=total_books,
            message=f"사용자 {user_id}의 {total_books}개 도서 정보를 조회했습니다."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"사용자 {user_id} 문서 상세 정보 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"사용자 {user_id} 문서 상세 정보 조회 중 오류가 발생했습니다: {str(e)}"
        )
