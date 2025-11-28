import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from functools import wraps
import json
import yaml

# 설정 파일 읽기
def load_config():
    """
    로깅 설정 파일을 로드하는 함수
    
    Returns:
        dict: 로깅 설정
    """
    config_path = os.path.join(os.path.dirname(__file__), 'logging_config.yaml')
    
    # 설정 파일이 없으면 기본 설정 반환
    if not os.path.exists(config_path):
        return {
            'use_elasticsearch': False,
            'elasticsearch': {
                'host': 'localhost',
                'port': 9200,
                'username': '',
                'password': ''
            },
            'log_dir': 'logs'
        }
    
    # YAML 설정 파일 읽기 (UTF-8 인코딩 지정)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_logger(name, log_dir='logs'):
    """
    로거 설정을 위한 함수
    
    Args:
        name (str): 로거 이름
        log_dir (str): 로그 저장 디렉토리 경로
        
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 설정 파일 로드
    config = load_config()
    
    # 로그 디렉토리 설정
    log_dir = config.get('log_dir', log_dir)
    
    # 로그 디렉토리 생성 (예외 처리 추가)
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # 로그 하위 디렉토리 설정
        log_category = name.split('_')[0] if '_' in name else name
        log_category_dir = os.path.join(log_dir, log_category)
        
        if not os.path.exists(log_category_dir):
            os.makedirs(log_category_dir, exist_ok=True)
    except Exception as e:
        # 디렉토리 생성 실패 시 현재 디렉토리 사용
        print(f"로그 디렉토리 생성 실패: {str(e)}")
        log_category_dir = '.'
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 부모 로거로의 전파 방지 (중복 로그 방지)
    logger.propagate = False

    # 중복 핸들러 방지
    if not logger.handlers:
        try:
            # 날짜별 로그 파일명 설정
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = os.path.join(log_category_dir, f'{name}_{today}.log')
            
            # 파일 핸들러 설정 (최대 10MB, 백업 5개)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 포맷터 설정
            formatter = logging.Formatter(
                '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # 핸들러 추가
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"로그 파일 핸들러 생성 실패: {str(e)}")
        
        # 콘솔 핸들러 설정 (항상 작동)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_api_logger(name):
    """
    API 엔드포인트 로깅을 위한 데코레이터
    
    Args:
        name (str): 로거 이름
        
    Returns:
        function: 데코레이터 함수
    """
    logger = logging.getLogger(name)
    
    def decorator(func):
        @wraps(func)  # 원본 함수의 메타데이터 보존
        async def wrapper(request, *args, **kwargs):
            try:
                logger.info(f"{func.__name__} API 요청 수신")
                logger.debug(f"요청 데이터: {request}")
                
                result = await func(request, *args, **kwargs)
                
                logger.info(f"{func.__name__} API 응답 성공")
                logger.debug(f"응답 구조: {list(result.keys()) if isinstance(result, dict) else '비 딕셔너리 응답'}")
                
                return result
                
            except Exception as e:
                logger.error(f"{func.__name__} 처리 중 오류 발생: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator