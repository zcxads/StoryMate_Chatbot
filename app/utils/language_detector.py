"""
언어 감지 유틸리티

사용자의 쿼리에서 언어를 자동 감지합니다.
"""

import re
from typing import Literal

# 언어 타입 정의
LanguageType = Literal["ko", "en", "ja", "zh", "unknown"]

def detect_language(text: str) -> LanguageType:
    """
    텍스트에서 언어를 감지합니다.

    Args:
        text: 분석할 텍스트

    Returns:
        LanguageType: 감지된 언어 코드 (ko, en, ja, zh, unknown)
    """
    if not text or not text.strip():
        return "unknown"

    text = text.strip()

    # 각 언어별 문자 개수 카운트
    korean_chars = len(re.findall(r'[가-힣]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    japanese_chars = len(re.findall(r'[ぁ-んァ-ヶー]', text))  # 히라가나, 가타카나
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))  # 한자 (중국어/일본어 공통)

    # 총 문자 수 (공백 제외)
    total_chars = len(re.sub(r'\s+', '', text))

    # 각 언어별 비율 계산
    korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
    english_ratio = english_chars / total_chars if total_chars > 0 else 0
    japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

    # 언어 결정 로직 (우선순위 기반)
    # 1. 한국어 (한글이 10% 이상) - 한글은 고유 문자이므로 우선순위 높음
    if korean_ratio > 0.1:
        return "ko"

    # 2. 일본어 (히라가나/가타카나가 5% 이상) - 일본어 고유 문자
    if japanese_ratio > 0.05:
        return "ja"

    # 3. 중국어 (한자가 20% 이상이고 일본어/한국어 문자가 거의 없음)
    # 임계값을 낮춰서 중국어 감지 개선
    if chinese_ratio > 0.2 and japanese_ratio < 0.05 and korean_ratio < 0.05:
        return "zh"

    # 4. 영어 (영문자가 30% 이상)
    if english_ratio > 0.3:
        return "en"

    # 5. 절대 개수 기반 판단 (비율이 낮을 때)
    if korean_chars > 0:
        return "ko"
    if japanese_chars > 0:
        return "ja"
    if chinese_chars > 2:  # 한자가 3개 이상이면 중국어로 간주
        return "zh"
    if english_chars > 0:
        return "en"

    # 6. 기본값은 unknown
    return "unknown"


def get_language_name(lang_code: LanguageType) -> str:
    """
    언어 코드를 언어 이름으로 변환합니다.

    Args:
        lang_code: 언어 코드

    Returns:
        str: 언어 이름
    """
    language_names = {
        "ko": "Korean",
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "unknown": "Unknown"
    }
    return language_names.get(lang_code, "Unknown")


def format_language_instruction(lang_code: LanguageType) -> str:
    """
    언어별 추가 지시사항을 포맷팅합니다.

    Args:
        lang_code: 언어 코드

    Returns:
        str: 언어별 추가 지시사항
    """
    instructions = {
        "ko": """

========================================
CRITICAL LANGUAGE ENFORCEMENT
========================================
사용자 질문 언어: 한국어 (Korean)
답변 언어: 반드시 100% 한국어만 사용

【필수 준수 사항】
1. 전체 답변을 한국어로만 작성하세요
2. 검색된 문서가 다른 언어여도 한국어로 번역/요약하세요
3. 영어, 일본어, 중국어 등 다른 언어를 절대 혼용하지 마세요
4. 띄어쓰기와 맞춤법을 정확히 지켜주세요
5. 전문 용어도 가능하면 한국어로 설명하세요

잘못된 예시: "이 책의 main character는..."
올바른 예시: "이 책의 주인공은..."
========================================
""",
        "en": """

========================================
CRITICAL LANGUAGE ENFORCEMENT
========================================
User Query Language: English
Response Language: MUST BE 100% ENGLISH ONLY

【Mandatory Requirements】
1. Write the entire response in English only
2. Translate/summarize documents in other languages into English
3. Never mix Korean, Japanese, Chinese or other languages
4. Use proper grammar and punctuation
5. Explain technical terms in English when possible

Wrong Example: "The 주인공 of this book..."
Correct Example: "The main character of this book..."
========================================
""",
        "ja": """

========================================
CRITICAL LANGUAGE ENFORCEMENT
========================================
ユーザー質問言語: 日本語 (Japanese)
回答言語: 必ず100%日本語のみ使用

【必須遵守事項】
1. 回答全体を日本語のみで作成してください
2. 検索された文書が他の言語でも日本語に翻訳/要約してください
3. 英語、韓国語、中国語など他の言語を絶対に混用しないでください
4. 適切な助詞と文法を使用してください
5. 専門用語もできるだけ日本語で説明してください

間違った例: "この本の main character は..."
正しい例: "この本の主人公は..."
========================================
""",
        "zh": """

========================================
CRITICAL LANGUAGE ENFORCEMENT
========================================
用户提问语言: 中文 (Chinese)
回答语言: 必须100%仅使用中文

【必须遵守事项】
1. 整个回答必须只用中文撰写
2. 即使检索到的文档是其他语言，也要翻译/总结成中文
3. 绝对不要混用英语、韩语、日语等其他语言
4. 使用正确的标点符号和语法结构
5. 专业术语也尽可能用中文解释

错误示例: "这本书的 main character 是..."
正确示例: "这本书的主人公是..."
========================================
""",
        "unknown": ""
    }
    return instructions.get(lang_code, "")
