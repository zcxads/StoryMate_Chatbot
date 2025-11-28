import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

from app.config import settings
from app.logs.logger import setup_logger

logger = setup_logger('chat_history_manager')

# 환경 변수 로드
load_dotenv()

class ChatHistoryManager:
    """벡터 기반 대화 내용 저장 및 관리 클래스"""

    def __init__(self, embeddings=None):
        # OpenAI API KEY 체크
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.strip() == "":
            raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다. API 키를 등록하세요.")

        # OpenAI Embeddings 초기화
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Qdrant API KEY, URL 체크
        if not settings.QDRANT_URL or settings.QDRANT_URL.strip() == "":
            raise RuntimeError("QDRANT_URL이 .env에 없습니다. Qdrant 서버 주소를 등록하세요.")

        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=getattr(settings, "QDRANT_API_KEY", None)
        )

        # 모델과 컬렉션의 dimension 맞추기 (예시: text-embedding-3-small 1536)
        self._vector_size = 1536
        self._distance = "Cosine"

        self._chat_history: Dict[str, List[Dict[str, str]]] = {}

    def save_conversation_to_vector_db(self, user_id: str, query: str, response: str) -> bool:
        """대화 내용을 벡터 DB에 저장합니다."""
        try:
            collection_name = f"{user_id}_chat"
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if collection_name not in collection_names:
                # 컬렉션 없는 경우 자동 생성
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "size": self._vector_size,
                        "distance": self._distance
                    }
                )
                logger.info(f"✅ 컬렉션 생성 완료: {collection_name}")

            conversation_text = f"{query}\n{response}"
            conversation_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            conversation_doc = Document(
                page_content=conversation_text,
                metadata={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "query": query,
                    "response": response,
                    "timestamp": timestamp
                }
            )

            # ----- 임베딩 생성 - dimension, None 체크
            try:
                embedding = self.embeddings.embed_query(conversation_text)
                if embedding is None or len(embedding) != self._vector_size:
                    logger.error(f"임베딩 오류: 반환값 None 또는 dimension 불일치! embedding={embedding}")
                    return False
            except Exception as emb_error:
                logger.error(f"임베딩 함수 에러: {emb_error}")
                return False

            # Qdrant에 포인트 적재
            try:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            "id": conversation_id,
                            "vector": embedding,
                            "payload": conversation_doc.metadata
                        }
                    ]
                )
            except Exception as upsert_error:
                logger.error(f"Qdrant 인서트 에러: {upsert_error}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ 전체 벡터 DB 저장 실패: {e}")
            return False

    def add_to_chat_history(self, user_id: str, query: str, response: str) -> None:
        """대화 히스토리에 신규 대화를 추가+벡터 DB 적재"""
        logger.info(f"💾 사용자 ID: {user_id}")
        logger.info(f"💾 질문: {query[:50]}... ({len(query)} 문자)")
        logger.info(f"💾 응답: {response[:50]}... ({len(response)} 문자)")
        
        # 메모리에 저장
        if user_id not in self._chat_history:
            self._chat_history[user_id] = []
        self._chat_history[user_id].append({"query": query, "response": response})
        logger.info(f"✅ 메모리 저장 완료. 현재 사용자 대화 수: {len(self._chat_history[user_id])}")
        
        # 벡터 DB에 저장
        try:
            save_success = self.save_conversation_to_vector_db(user_id, query, response)
            if save_success:
                logger.info("✅ 벡터 DB 저장 완료")
            else:
                logger.error("❌ 벡터 DB 저장 실패")
        except Exception as e:
            logger.error(f"❌ 벡터 DB 저장 중 오류: {e}", exc_info=True)

    def get_conversation_by_index(self, user_id: str, index: int) -> Optional[Dict[str, str]]:
        """메모리에서 특정 인덱스의 대화를 가져옵니다.

        Args:
            user_id: 사용자 ID
            index: 대화 인덱스 (0-based, 음수는 뒤에서부터 -1=마지막)

        Returns:
            대화 딕셔너리 {"query": "...", "response": "..."} 또는 None
        """
        try:
            if user_id not in self._chat_history:
                logger.warning(f"사용자 {user_id}의 대화 기록이 없습니다.")
                return None

            user_history = self._chat_history[user_id]

            if not user_history:
                logger.warning(f"사용자 {user_id}의 대화 기록이 비어있습니다.")
                return None

            # 인덱스 범위 확인
            if abs(index) > len(user_history):
                logger.warning(f"인덱스 {index}가 범위를 벗어남 (총 {len(user_history)}개 대화)")
                return None

            conversation = user_history[index]
            logger.info(f"✅ 인덱스 {index} 대화 조회 성공 (총 {len(user_history)}개 중)")

            return conversation

        except Exception as e:
            logger.error(f"대화 인덱스 조회 실패: {e}")
            return None

    def get_conversation_count(self, user_id: str) -> int:
        """사용자의 총 대화 개수를 반환합니다."""
        if user_id not in self._chat_history:
            return 0
        return len(self._chat_history[user_id])

    def retrieve_vector_based_memory(self, user_id: str, query: str, search_params) -> List[Any]:
        """벡터 기반, 유사한 대화 검색"""
        try:
            chat_collection_name = f"{user_id}_chat"
            collection_names = [col.name for col in self.qdrant_client.get_collections().collections]
            if chat_collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=chat_collection_name,
                    vectors_config={
                        "size": self._vector_size,
                        "distance": self._distance
                    }
                )
            try:
                query_embedding = self.embeddings.embed_query(query)
                if query_embedding is None or len(query_embedding) != self._vector_size:
                    logger.error(f"쿼리 임베딩 오류: 반환값 None 또는 dimension 불일치! embedding={query_embedding}")
                    return []
            except Exception as emb_error:
                logger.error(f"쿼리 임베딩 함수 에러: {emb_error}")
                return []

            effective_score_threshold = min(getattr(search_params, "score_threshold", 0.3), 0.3)
            search_results = []
            try:
                search_results = self.qdrant_client.search(
                    collection_name=chat_collection_name,
                    query_vector=query_embedding,
                    limit=getattr(search_params, "k", 10) * 3,
                    with_payload=True,
                    score_threshold=effective_score_threshold
                )
            except Exception as search_error:
                logger.error(f"Qdrant 검색 에러: {search_error}")
                return []

            # 검색 결과 Document 변환 및 정렬(유사도 역순)
            filtered_conversations = []
            for result in search_results:
                payload = result.payload
                if payload:
                    payload["similarity_score"] = result.score
                    doc = Document(
                        page_content=f"{payload.get('query', '')}\n{payload.get('response', '')}",
                        metadata=payload
                    )
                    filtered_conversations.append(doc)

            sorted_conversations = sorted(
                filtered_conversations,
                key=lambda x: x.metadata.get("similarity_score", 0),
                reverse=True
            )
            return sorted_conversations
        except Exception as e:
            logger.error(f"❌ 벡터 기반 메모리 검색 실패: {e}")
            return []
