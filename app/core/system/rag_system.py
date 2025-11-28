import time
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langchain_core.documents import Document

from app.states import MultiturnRAGState
from app.core.retrieval.vector_store import VectorStore
from app.core.chat.chat_history_manager import ChatHistoryManager
from app.core.chat.intent_analyzer import IntentAnalyzer
from app.core.document.document_context import DocumentContextManager
from app.core.document.document_loader import DocumentLoader
from app.core.retrieval.retriever_manager import RetrieverManager, SearchParams
from app.core.llm.llm_provider import LLMProvider
from app.core.chat.intent_analyzer_agent import IntentAnalyzerAgent
from app.core.document.context_manager_agent import ContextManagerAgent
from app.core.retrieval.document_retriever_agent import DocumentRetrieverAgent
from app.core.llm.answer_generator_agent import AnswerGeneratorAgent
from app.config import settings
from app.logs.logger import setup_logger

logger = setup_logger('rag_system')

# LangSmith tracing helper functions
def process_inputs(inputs: dict) -> dict:
    """Limit the inputs displayed in LangSmith to only the query."""
    return {
        "query": inputs.get("query", "")
    }

def process_outputs(outputs: dict) -> dict:
    """Limit the outputs displayed in LangSmith to only the answer."""
    return {
        "answer": outputs.get("answer", "")
    }

class MultiturnRAGSystem:
    """LangGraph-based multi-turn RAG system"""
    
    def __init__(self):
        """Initializes the RAG system."""
        from langchain_openai import OpenAIEmbeddings
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize components
        self.vector_store = VectorStore(self.embeddings)
        self.chat_history_manager = ChatHistoryManager(self.embeddings)
        self.document_loader = DocumentLoader()
        
        # Initialize agents
        self.intent_analyzer_agent = IntentAnalyzerAgent()
        self.context_manager_agent = ContextManagerAgent(self)
        self.document_retriever_agent = DocumentRetrieverAgent(self)
        self.answer_generator_agent = AnswerGeneratorAgent(self)
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Creates the LangGraph workflow with conditional routing based on intent."""
        workflow = StateGraph(MultiturnRAGState)

        # Add nodes
        workflow.add_node("intent_analyzer", self.intent_analyzer_agent.analyze_intent)
        workflow.add_node("context_manager", self.context_manager_agent.manage_context)
        workflow.add_node("document_retriever", self.document_retriever_agent.retrieve_documents)
        workflow.add_node("follow_up_handler", self.handle_follow_up_request)
        workflow.add_node("answer_generator", self.answer_generator_agent.generate_answer)
        workflow.add_node("fallback_to_general_chat", self.fallback_to_general_chat)

        # Define edges
        workflow.set_entry_point("intent_analyzer")
        workflow.add_conditional_edges(
            "intent_analyzer",
            self.should_retrieve_documents,
            {
                "retrieve": "context_manager",
                "handle_follow_up": "follow_up_handler",
                "answer_direct": "answer_generator"
            }
        )
        workflow.add_edge("context_manager", "document_retriever")
        workflow.add_conditional_edges(
            "document_retriever",
            self.check_documents_retrieved,
            {
                "has_documents": "answer_generator",
                "no_documents": "fallback_to_general_chat"
            }
        )
        workflow.add_edge("fallback_to_general_chat", "answer_generator")
        workflow.add_edge("follow_up_handler", "answer_generator")
        workflow.add_edge("answer_generator", END)

        return workflow.compile()

    def should_retrieve_documents(self, state: MultiturnRAGState) -> str:
        """Determines the next node based on the intent."""
        intent = state.get("intent")
        logger.info(f" Routing decision: intent='{intent}'")

        if intent == "follow_up_summary":
            return "handle_follow_up"
        elif intent == "general_chat":
            # 일반 채팅은 문서 검색과 컨텍스트 관리 없이 바로 답변 생성
            return "answer_direct"
        else:
            return "retrieve"

    def check_documents_retrieved(self, state: MultiturnRAGState) -> str:
        """문서 검색 후 문서가 있는지 확인하고, 없으면 일반 채팅으로 폴백"""
        retrieved_documents = state.get("retrieved_documents", [])
        error = state.get("error")
        intent = state.get("intent")

        # 에러가 있거나 문서가 없으면 일반 채팅으로 폴백
        if error or not retrieved_documents or len(retrieved_documents) == 0:
            logger.warning(f"⚠️ 문서 검색 실패 또는 문서 없음 (intent: {intent}). 일반 채팅으로 폴백")
            return "no_documents"

        logger.info(f"✅ 문서 검색 성공: {len(retrieved_documents)}개 문서")
        return "has_documents"

    def fallback_to_general_chat(self, state: MultiturnRAGState) -> MultiturnRAGState:
        """문서가 없을 때 일반 채팅으로 폴백"""
        logger.info("💬 문서가 없어 일반 채팅 모드로 전환")

        # intent를 general_chat으로 변경하고 에러 메시지 제거
        state["intent"] = "general_chat"
        state["error"] = None
        state["retrieved_documents"] = []

        return state

    def handle_follow_up_request(self, state: MultiturnRAGState) -> Dict[str, Any]:
        """Handles follow-up requests by analyzing LLM-detected reference index."""
        user_id = state.get('user_id', '')
        query = state.get('query', '')
        reference_index = state.get('reference_index')
        reference_type = state.get('reference_type')

        logger.info("🔄 follow_up_summary 처리 시작")
        logger.info(f"🔍 사용자 질문: {query}")

        # LLM이 분석한 reference 정보 사용
        if reference_index is None:
            logger.warning("⚠️ reference_index가 None, 기본값 -1 (마지막 대화) 사용")
            reference_index = -1
            reference_type = "last"

        logger.info(f"🎯 LLM 분석 결과: reference_index={reference_index}, reference_type={reference_type}")

        # 대화 개수 확인
        total_conversations = self.chat_history_manager.get_conversation_count(user_id)
        logger.info(f"📊 총 대화 개수: {total_conversations}")

        if total_conversations == 0:
            logger.warning("❌ 대화 기록이 없음")
            fallback_doc = Document(
                page_content="참조할 이전 대화가 없습니다.",
                metadata={"source": "session_memory", "type": "follow_up_summary", "index": None}
            )
            return {"retrieved_documents": [fallback_doc]}

        # 인덱스 범위 검증
        if reference_index >= 0 and reference_index >= total_conversations:
            logger.warning(f"⚠️ 인덱스 {reference_index}가 범위 초과 (총 {total_conversations}개), 마지막 대화로 폴백")
            reference_index = -1
            reference_type = "last (범위 초과로 폴백)"

        logger.info(f"🎯 최종 선택: {reference_type} 대화 (인덱스: {reference_index})")

        # 대화 가져오기
        conversation = self.chat_history_manager.get_conversation_by_index(user_id, reference_index)

        if not conversation:
            logger.error(f"❌ 인덱스 {reference_index} 대화를 찾을 수 없음")
            fallback_doc = Document(
                page_content="해당 대화를 찾을 수 없습니다.",
                metadata={"source": "session_memory", "type": "follow_up_summary", "index": reference_index}
            )
            return {"retrieved_documents": [fallback_doc]}

        # 질문과 답변 추출
        previous_query = conversation.get("query", "")
        previous_response = conversation.get("response", "")

        logger.info(f"✅ {reference_type} 대화 조회 성공")
        logger.info(f"   📝 질문: {previous_query[:50]}...")
        logger.info(f"   💬 답변: {previous_response[:50]}...")

        # 전체 대화 내용 구성
        conversation_content = f"[{reference_type} 질문]\n{previous_query}\n\n[답변]\n{previous_response}"

        follow_up_doc = Document(
            page_content=conversation_content,
            metadata={
                "source": "session_memory",
                "type": "follow_up_summary",
                "index": reference_index,
                "index_type": reference_type,
                "previous_query": previous_query,
                "previous_response": previous_response
            }
        )

        return {"retrieved_documents": [follow_up_doc]}
    
    # --- Document management methods ---
    def load_documents_from_pages(self, pages: List[Any], user_id: str, book_id: int) -> List:
        return self.document_loader.load_documents_from_pages(pages, user_id, book_id)

    def create_chunks(self, documents: List) -> List:
        return self.document_loader.create_chunks(documents)

    def create_vector_db(self, chunks: List, user_id: str, book_id: int) -> bool:
        return self.vector_store.create_vector_db(chunks, user_id, book_id)

    def get_user_vector_db(self, user_id: str, collection_name: str):
        return self.vector_store.get_user_vector_db(user_id, collection_name)

    def get_user_document_detail(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self.vector_store.get_user_document_detail(user_id)

    
    @traceable(
        name="RAG System",
        run_type="chain",
        process_inputs=process_inputs,
        process_outputs=process_outputs
    )
    async def retrieve_and_generate(
        self, query: str, user_id: str, model_name:str = None
    ) -> Dict[str, Any]:
        """Generates a response and saves the conversation to history."""
        start_time = time.time()
        
        try:
            conversation_history = self._load_conversation_history(user_id, query)
            
            initial_state = MultiturnRAGState(
                user_id=user_id, query=query, conversation_history=conversation_history,
                intent=None, search_context=None, retrieved_documents=None, answer=None
            )
            
            query_preview = query[:50] + "..." if len(query) > 50 else query
            logger.info(f" Starting multi-turn RAG for user: {user_id}, query: {query_preview}...")
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(" Final state analysis:")
            logger.info(f"  Intent: {final_state.get('intent')}")
            
            answer = final_state.get("answer", "")
            
            # 대화 기록 저장은 Answer Generator에서 이미 처리됨 (중복 저장 방지)
            if answer:
                logger.info(f"✅ 답변 생성 완료 (길이: {len(answer)} 문자)")

            response = {"answer": answer, "intent": final_state.get("intent")}
            
            execution_time = time.time() - start_time
            logger.info(f" Multi-turn RAG finished in {execution_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logger.error(f"Multi-turn RAG failed: {e}", exc_info=True)
            return {"answer": "Sorry, an error occurred while generating a response.", "intent": "error"}
    
    def _load_conversation_history(self, user_id: str, query: str) -> List[Dict[str, str]]:
        """Loads conversation history from the vector store."""
        try:
            search_params = SearchParams(k=10, score_threshold=0.3, timeout=10)
            chat_history_docs = self.chat_history_manager.retrieve_vector_based_memory(user_id, query, search_params)
            
            logger.info(f"🔍 대화 기록 문서 개수: {len(chat_history_docs)}")
            
            conversation_history = []
                        
            # 최종 대화 기록 구조 확인
            if conversation_history:
                last_conv = conversation_history[-1]
                logger.info(f"🎯 마지막 대화 구조: {list(last_conv.keys())}")
                logger.info(f"🎯 마지막 대화 assistant 내용 (100자): {last_conv.get('assistant', 'None')[:100]}...")
            
            return conversation_history
            
        except Exception as e:
            logger.error(f"❌ 대화 기록 로딩 실패: {e}", exc_info=True)
            return []