# 네이버 검색광고 키워드 챗봇

이 프로젝트는 네이버 검색광고 API와 GPT를 활용하여 키워드 정보를 제공하는 챗봇입니다.

## 기능

- 네이버 검색광고 키워드 정보 조회
- GPT를 활용한 자연스러운 대화형 응답
- 키워드 입찰가, 상태 등 상세 정보 제공

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
`.env` 파일을 생성하고 다음 정보를 입력하세요:
```
OPENAI_API_KEY=your_openai_api_key
NAVER_API_KEY=your_naver_api_key
NAVER_API_SECRET=your_naver_api_secret
NAVER_CUSTOMER_ID=your_customer_id
```

## 실행 방법

```bash
python main.py
```

서버가 실행되면 `http://localhost:8000`에서 API를 사용할 수 있습니다.

## API 사용 예시

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "키워드 현재 입찰가를 알려줘", "keyword_id": "your_keyword_id"}'
```

## 주의사항

- 네이버 검색광고 API 키와 시크릿은 네이버 검색광고 센터에서 발급받아야 합니다.
- OpenAI API 키는 OpenAI 웹사이트에서 발급받을 수 있습니다.
- 모든 API 키는 안전하게 보관하고 절대 공개되지 않도록 주의하세요. 