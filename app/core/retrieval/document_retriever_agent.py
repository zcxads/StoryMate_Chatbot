from typing import Dict, Any
from app.core.llm.prompt_manager import PromptManager
from app.core.document.document_context import DocumentContextManager
from app.core.retrieval.vector_store import VectorStore
from app.core.retrieval.retriever_manager import RetrieverManager, SearchParams
from app.core.chat.chat_history_manager import ChatHistoryManager
from app.logs.logger import setup_logger

logger = setup_logger('document_retriever_agent')

class DocumentRetrieverAgent:
    """관련 문서를 검색하는 에이전트"""

    def __init__(self, rag_system):
        """문서 검색 에이전트 초기화"""
        self.rag_system = rag_system
        self.prompt_manager = PromptManager()

    async def retrieve_documents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """의도와 컨텍스트에 따라 관련 문서를 검색합니다."""
        import time

        start_time = time.time()
        try:
            user_id = state["user_id"]
            query = state["query"]
            intent = state.get("intent", "general_chat")
            use_unified = state.get("use_unified_collection", False)  # 통합 컬렉션 사용 여부

            # 일반 채팅: 문서 검색 없이 바로 진행
            if intent == "general_chat":
                logger.info("💬 일반 채팅: 문서 검색 건너뜀")
                state["retrieved_documents"] = []
                return state
            
            # 사용자 문서 컨텍스트 가져오기
            document_context_manager = DocumentContextManager(self.rag_system)
            try:
                document_context = document_context_manager.get_all_user_documents_context(user_id, query, intent)
                
                if not document_context or document_context == "사용자가 업로드한 문서가 없습니다.":
                    logger.info(f"📝 사용자 {user_id}의 문서가 없습니다.")
                    state["error"] = f"죄송합니다. 사용자 {user_id}님의 검색 가능한 문서가 없습니다. 먼저 문서를 업로드해주세요."
                    return state
                    
            except Exception as e:
                logger.error(f"문서 컨텍스트 가져오기 실패: {e}")
                state["error"] = f"죄송합니다. 사용자 {user_id}님의 문서 정보를 가져올 수 없습니다."
                return state
            
            # 사용자 벡터 데이터베이스 가져오기
            vector_store = VectorStore(self.rag_system.embeddings)
            try:
                # 올바른 컬렉션 이름 생성
                collection_name = vector_store.get_user_collection_name(user_id)
                user_collection = vector_store.get_vector_db_for_user(user_id, collection_name)
                
                if not user_collection:
                    logger.info(f"📝 사용자 {user_id}의 벡터 데이터베이스가 없습니다. 문서를 먼저 업로드해주세요.")
                    state["error"] = f"죄송합니다. 사용자 {user_id}님의 벡터 데이터베이스에 접근할 수 없습니다. 먼저 문서를 업로드해주세요."
                    return state
                    
            except Exception as e:
                logger.error(f"벡터 데이터베이스 접근 실패: {e}")
                state["error"] = f"죄송합니다. 사용자 {user_id}님의 벡터 데이터베이스에 접근할 수 없습니다."
                return state
            
            # 문서 목록 질문
            if intent == "document_list":
                # 문서 목록 질문: book_id별로 그룹화
                logger.info(f"📚 문서 목록 질문: book_id별 그룹화")

                try:
                    # 사용자 문서 상세 정보 가져오기
                    user_document_detail = self.rag_system.get_user_document_detail(user_id)

                    if not user_document_detail or not user_document_detail.get("books"):
                        logger.info(f"📝 사용자 {user_id}의 문서가 없습니다.")
                        state["error"] = f"죄송합니다. 사용자 {user_id}님의 검색 가능한 문서가 없습니다."
                        return state

                    books = user_document_detail["books"]
                    logger.info(f"📚 총 {len(books)}개의 문서 발견")

                    # 각 문서의 샘플 텍스트만 추출
                    from langchain_core.documents import Document

                    document_samples = []
                    for idx, book in enumerate(books):
                        book_id = book.get("bookKey", "unknown")
                        chunks = book.get("chunks", [])

                        if not chunks:
                            logger.warning(f"📝 book_id {book_id}: 청크가 없음")
                            continue

                        # 첫 5개 청크에서 샘플 텍스트 추출
                        sample_texts = []
                        for chunk in chunks[:5]:
                            text = chunk.get("text", "").strip()
                            if text:
                                sample_texts.append(text[:500])  # 각 청크에서 최대 500자

                        if not sample_texts:
                            logger.warning(f"📝 book_id {book_id}: 텍스트가 비어있음")
                            continue

                        # 샘플 텍스트 결합
                        combined_sample = "\n\n".join(sample_texts)

                        # Document 객체 생성 (제목 생성은 답변 생성 LLM에 위임)
                        document_doc = Document(
                            page_content=combined_sample,
                            metadata={
                                "source": "document_list",
                                "book_id": book_id,
                                "book_index": idx + 1,
                                "total_pages": len(chunks),
                                "intent": "document_list"
                            }
                        )
                        document_samples.append(document_doc)
                        logger.info(f"✅ book_id {book_id} - 샘플 텍스트 추출 ({len(combined_sample)} 문자)")

                    if not document_samples:
                        logger.warning("처리할 문서가 없습니다.")
                        state["error"] = "처리할 문서가 없습니다."
                        return state

                    state["retrieved_documents"] = document_samples
                    logger.info(f"✅ {len(document_samples)}개 문서 샘플 추출 완료")

                except Exception as e:
                    logger.error(f"문서 목록 처리 실패: {e}")
                    state["error"] = f"죄송합니다. 문서 목록을 가져올 수 없습니다."
                    return state

            else:
                # 세부 질문: 하이브리드 검색 사용 (벡터 + BM25)
                logger.info("🔍 세부 질문: 하이브리드 검색 사용 (벡터 + BM25)")
                
                # 하이브리드 검색을 위한 검색기 생성
                retriever_manager = RetrieverManager(self.rag_system.embeddings)
                try:
                    # 하이브리드 검색을 위한 최적화된 검색 파라미터 생성
                    # - 벡터 검색: 시맨틱 유사도 기반
                    # - BM25 검색: 키워드 매칭 기반
                    # - 앙상블 가중치: 벡터 70%, BM25 30%
                    search_params = SearchParams(
                        k=15,  # 하이브리드 검색을 위해 더 많은 후보 검색
                        score_threshold=0.25,  # 하이브리드 검색을 위해 임계값 낮춤
                        timeout=30
                    )
                    
                    # 하이브리드 retriever 생성 (벡터 + BM25)
                    hybrid_retriever = retriever_manager.create_hybrid_retriever(
                        user_collection, 
                        search_params, 
                        user_id=user_id
                    )
                    
                    if not hybrid_retriever:
                        logger.warning("하이브리드 retriever 생성 실패, 기본 벡터 retriever 사용")
                        # 폴백: 기본 벡터 retriever 사용
                        retriever = retriever_manager.create_retriever(user_collection, search_params)
                        if not retriever:
                            state["error"] = f"죄송합니다. 사용자 {user_id}님의 검색기를 생성할 수 없습니다."
                            return state
                        hybrid_retriever = retriever
                        
                except Exception as e:
                    logger.error(f"하이브리드 검색기 생성 실패: {e}")
                    state["error"] = f"죄송합니다. 사용자 {user_id}님의 검색기에 접근할 수 없습니다."
                    return state
                
                # 동적 k값 계산 (하이브리드 검색 최적화)
                # - 질문 길이에 따른 동적 k값 조정
                # - 하이브리드 검색을 위해 더 많은 후보 검색
                base_k = len(query.split()) * 3  # 질문 길이의 3배
                k_value = min(max(base_k, 8), 25)  # 최소 8개, 최대 25개
                
                logger.info(f"🔍 하이브리드 검색 파라미터: k={k_value}, 질문 길이={len(query.split())}")
                
                # 하이브리드 문서 검색
                retrieved_documents = hybrid_retriever.get_relevant_documents(query, k=k_value)
                
                # 검색 결과 로깅 및 중복 제거
                if retrieved_documents:
                    logger.info(f"✅ 하이브리드 검색 완료: {len(retrieved_documents)}개 문서 검색됨")
                    
                    # 중복 제거 (동일한 내용의 문서 제거)
                    unique_documents = []
                    seen_contents = set()
                    
                    for doc in retrieved_documents:
                        # 문서 내용의 해시 생성 (처음 200자 기준)
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_contents:
                            unique_documents.append(doc)
                            seen_contents.add(content_hash)
                    
                    # 중복 제거 결과 로깅
                    removed_count = len(retrieved_documents) - len(unique_documents)
                    if removed_count > 0:
                        logger.info(f"🧹 중복 제거: {removed_count}개 문서 제거됨")
                    
                    logger.info(f"📊 최종 검색 결과: {len(unique_documents)}개 고유 문서")
                    state["retrieved_documents"] = unique_documents
                    
                else:
                    logger.warning("⚠️ 하이브리드 검색 결과가 없습니다.")
                    state["retrieved_documents"] = []
                    
                    # 폴백: 대화 기록에서 검색 시도
                    logger.info("폴백: 문서 없음 → 대화 기록에서 검색 시도")
                    try:
                        chat_history_manager = ChatHistoryManager(self.rag_system.embeddings)
                        fallback_search_params = SearchParams(k=10, score_threshold=0.3, timeout=30)
                        memory_conversations = chat_history_manager.retrieve_vector_based_memory(
                            user_id, query, fallback_search_params
                        )
                        
                        if memory_conversations:
                            logger.info(f"폴백 성공: 대화 기록에서 {len(memory_conversations)}개 검색됨")
                            state["memory_conversations"] = memory_conversations
                            state["fallback_to_memory"] = True
                        else:
                            logger.warning("대화 기록에서도 관련 내용을 찾을 수 없습니다.")
                            state["memory_conversations"] = []
                            
                    except Exception as fallback_error:
                        logger.error(f"폴백 검색 실패: {fallback_error}")
                        state["memory_conversations"] = []
            
            # 검색 성능 로깅
            search_time = time.time() - start_time
            logger.info(f"⏱️ 문서 검색 완료: {search_time:.2f}초")
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"문서 검색 실패 ({search_time:.2f}초): {e}")
            state["error"] = f"죄송합니다. 사용자 {user_id}님의 검색기에 접근할 수 없습니다."
        
        return state 