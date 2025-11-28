import os
import pytest
import requests

# 서버 주소를 환경변수 또는 pytest 옵션으로 지정
def get_server_url(pytestconfig):
    url = pytestconfig.getoption("server_url") or os.getenv("SERVER_URL")
    if url:
        return url.rstrip("/")
    return "http://175.209.197.38:8007"

@pytest.fixture(scope="session")
def server_url(pytestconfig):
    return get_server_url(pytestconfig)

def test_health_check(server_url):
    """헬스체크 엔드포인트 테스트"""
    # 실제 서버 엔드포인트에 맞게 경로 수정
    response = requests.get(f"{server_url}/api/v1/health")
    assert response.status_code == 200, f"/api/v1/health 응답 코드: {response.status_code}, 내용: {response.text}"

def test_get_models(server_url):
    """지원 모델 조회 테스트"""
    response = requests.get(f"{server_url}/api/v1/models")
    assert response.status_code == 200, f"/api/v1/models 응답 코드: {response.status_code}, 내용: {response.text}"

def test_upload_documents(server_url):
    """문서 업로드 테스트"""
    test_data = {
        "user_id": "0",
        "book_id": 1,
        "pages": [
            {
                "pageKey": 1,
                "text": "테스트 문서입니다."
            }
        ]
    }
    response = requests.post(f"{server_url}/api/v1/documents/upload", json=test_data)
    assert response.status_code == 200, f"/api/v1/documents/upload 응답 코드: {response.status_code}, 내용: {response.text}"

def test_chat_endpoint(server_url):
    """채팅 엔드포인트 테스트"""
    test_data = {
        "user_id": "0",
        "query": "안녕하세요",
        "model": "gemini-2.0-flash"
    }
    response = requests.post(f"{server_url}/api/v1/chat", json=test_data)
    assert response.status_code in [200, 500], f"/api/v1/chat 응답 코드: {response.status_code}, 내용: {response.text}"

def test_get_user_documents(server_url):
    """사용자 문서 조회 테스트"""
    response = requests.get(f"{server_url}/api/v1/documents/user/0")
    assert response.status_code in [200, 404], f"/api/v1/documents/user/0 응답 코드: {response.status_code}, 내용: {response.text}" 