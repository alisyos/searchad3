from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import openai
import requests
import os
import time
import logging
import hashlib
import hmac
import base64
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")

# 네이버 검색광고 API 설정
NAVER_API_URL = "https://api.searchad.naver.com"
NAVER_API_KEY = os.getenv("NAVER_API_KEY")
NAVER_API_SECRET = os.getenv("NAVER_API_SECRET")
NAVER_CUSTOMER_ID = os.getenv("NAVER_CUSTOMER_ID")

if not all([NAVER_API_KEY, NAVER_API_SECRET, NAVER_CUSTOMER_ID]):
    logger.error("네이버 API 관련 환경 변수가 설정되지 않았습니다.")

class ChatRequest(BaseModel):
    message: str
    keyword_id: str = None

def generate_signature(timestamp: str, method: str, uri: str, secret_key: str) -> str:
    """네이버 API 요청에 필요한 서명을 생성합니다."""
    # URI에서 쿼리 파라미터 제거
    base_uri = uri.split('?')[0]
    message = f"{timestamp}.{method}.{base_uri}"
    
    # 디버깅을 위한 로그 추가
    logger.info(f"서명 생성에 사용된 값들:")
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"method: {method}")
    logger.info(f"base_uri: {base_uri}")
    logger.info(f"message: {message}")
    
    secret_key_bytes = bytes(secret_key, 'UTF-8')
    message_bytes = bytes(message, 'UTF-8')
    
    hash = hmac.new(secret_key_bytes, message_bytes, hashlib.sha256)
    signature = base64.b64encode(hash.digest()).decode('utf-8')
    
    logger.info(f"생성된 서명: {signature}")
    return signature

def get_keyword_info(keyword_id: str):
    """네이버 검색광고 API를 통해 키워드 정보를 가져옵니다."""
    try:
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        uri = f"/ncc/keywords/{keyword_id}"
        
        signature = generate_signature(timestamp, method, uri, NAVER_API_SECRET)
        
        headers = {
            "X-Timestamp": timestamp,
            "X-API-KEY": NAVER_API_KEY,
            "X-Customer": NAVER_CUSTOMER_ID,
            "X-Signature": signature
        }
        
        logger.info(f"네이버 API 요청 시작: {keyword_id}")
        logger.info(f"요청 헤더: {headers}")
        
        response = requests.get(
            f"{NAVER_API_URL}{uri}",
            headers=headers
        )
        
        if response.status_code != 200:
            logger.error(f"네이버 API 오류: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"키워드 정보를 가져오는데 실패했습니다: {response.text}")
        
        return response.json()
    except Exception as e:
        logger.error(f"키워드 정보 조회 중 오류 발생: {str(e)}")
        raise

def update_keyword_bid(keyword_id: str, bid_amount: int, use_group_bid: bool = False):
    """네이버 검색광고 API를 통해 키워드 입찰가를 수정합니다."""
    try:
        timestamp = str(int(time.time() * 1000))
        method = "PUT"
        uri = f"/ncc/keywords/{keyword_id}?fields=bidAmt"
        
        signature = generate_signature(timestamp, method, uri, NAVER_API_SECRET)
        
        headers = {
            "X-Timestamp": timestamp,
            "X-API-KEY": NAVER_API_KEY,
            "X-Customer": NAVER_CUSTOMER_ID,
            "X-Signature": signature,
            "Content-Type": "application/json"
        }
        
        # 키워드 정보를 먼저 가져와 그룹 ID를 얻습니다
        keyword_info = get_keyword_info(keyword_id)
        
        data = {
            "nccKeywordId": keyword_id,
            "nccAdgroupId": keyword_info["nccAdgroupId"],
            "bidAmt": bid_amount,
            "useGroupBidAmt": use_group_bid
        }
        
        logger.info(f"키워드 입찰가 업데이트 요청: {keyword_id}, {bid_amount}원")
        logger.info(f"요청 헤더: {headers}")
        logger.info(f"요청 데이터: {data}")
        
        response = requests.put(
            f"{NAVER_API_URL}{uri}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            logger.error(f"네이버 API 오류: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"키워드 입찰가 수정에 실패했습니다: {response.text}")
        
        return response.json()
    except Exception as e:
        logger.error(f"키워드 입찰가 수정 중 오류 발생: {str(e)}")
        raise

def update_keyword_status(keyword_id: str, enabled: bool):
    """네이버 검색광고 API를 통해 키워드 상태(활성화/일시중지)를 수정합니다."""
    try:
        timestamp = str(int(time.time() * 1000))
        method = "PUT"
        uri = f"/ncc/keywords/{keyword_id}?fields=userLock"
        
        signature = generate_signature(timestamp, method, uri, NAVER_API_SECRET)
        
        headers = {
            "X-Timestamp": timestamp,
            "X-API-KEY": NAVER_API_KEY,
            "X-Customer": NAVER_CUSTOMER_ID,
            "X-Signature": signature,
            "Content-Type": "application/json"
        }
        
        # 키워드 정보를 먼저 가져와 그룹 ID를 얻습니다
        keyword_info = get_keyword_info(keyword_id)
        
        # userLock이 true면 PAUSED, false면 ENABLED 상태가 됩니다
        data = {
            "nccKeywordId": keyword_id,
            "nccAdgroupId": keyword_info["nccAdgroupId"],
            "userLock": not enabled  # enabled가 true면 userLock은 false
        }
        
        status = "활성화" if enabled else "일시중지"
        logger.info(f"키워드 상태 업데이트 요청: {keyword_id}, {status}")
        logger.info(f"요청 헤더: {headers}")
        logger.info(f"요청 데이터: {data}")
        
        response = requests.put(
            f"{NAVER_API_URL}{uri}",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            logger.error(f"네이버 API 오류: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"키워드 상태 수정에 실패했습니다: {response.text}")
        
        return response.json()
    except Exception as e:
        logger.error(f"키워드 상태 수정 중 오류 발생: {str(e)}")
        raise

def analyze_user_command(message: str, keyword_info: dict = None):
    """사용자 메시지를 분석하여 수행할 작업을 결정합니다."""
    try:
        # GPT에 전달할 프롬프트
        system_prompt = """
        당신은 네이버 검색광고 키워드 설정을 관리하는 AI 비서입니다.
        사용자의 메시지를 분석하여 수행해야 할 작업을 결정해야 합니다.
        
        가능한 작업은 다음과 같습니다:
        1. 입찰가 수정 - 예: "입찰가를 700원으로 변경해줘"
        2. 상태 변경 - 예: "키워드를 활성화해줘", "키워드를 일시중지해줘"
        3. 정보 조회 - 예: "현재 입찰가는 얼마야?"
        
        다음 형식으로 응답해주세요:
        ```json
        {
            "action": "update_bid" 또는 "update_status" 또는 "get_info" 또는 "unknown",
            "bid_amount": 123,  # 입찰가 수정 시에만
            "status": true/false,  # 상태 변경 시에만 (true: 활성화, false: 일시중지)
            "explanation": "사용자의 요청에 대한 이해 설명"
        }
        ```
        """
        
        # 키워드 정보가 있는 경우 프롬프트에 추가
        if keyword_info:
            system_prompt += f"\n현재 키워드 정보:\n{keyword_info}"
        
        # GPT API 호출
        logger.info("명령 분석을 위한 GPT API 호출")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            response_format={"type": "json_object"}
        )
        
        # JSON 응답 파싱
        analysis = response.choices[0].message.content
        logger.info(f"분석 결과: {analysis}")
        
        import json
        return json.loads(analysis)
    except Exception as e:
        logger.error(f"사용자 명령 분석 중 오류 발생: {str(e)}")
        return {"action": "unknown", "explanation": "명령을 분석하는 중 오류가 발생했습니다."}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>네이버 검색광고 키워드 챗봇</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
            .input-group { margin-bottom: 10px; }
            input[type="text"] { width: 100%; padding: 8px; margin: 5px 0; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            #response { margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; min-height: 100px; }
            .examples { margin-top: 20px; }
            .example-item { cursor: pointer; color: #0066cc; margin-bottom: 5px; }
            .example-item:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>네이버 검색광고 키워드 챗봇</h1>
        <div class="chat-container">
            <div class="input-group">
                <label for="message">메시지:</label>
                <input type="text" id="message" placeholder="질문을 입력하세요">
            </div>
            <div class="input-group">
                <label for="keyword_id">키워드 ID:</label>
                <input type="text" id="keyword_id" placeholder="키워드 ID를 입력하세요">
            </div>
            <button onclick="sendMessage()">전송</button>
            <div id="response"></div>
            
            <div class="examples">
                <h3>예시 질문:</h3>
                <div class="example-item" onclick="setExample('현재 입찰가를 알려줘')">현재 입찰가를 알려줘</div>
                <div class="example-item" onclick="setExample('입찰가를 700원으로 수정해줘')">입찰가를 700원으로 수정해줘</div>
                <div class="example-item" onclick="setExample('키워드를 활성화해줘')">키워드를 활성화해줘</div>
                <div class="example-item" onclick="setExample('키워드를 일시중지해줘')">키워드를 일시중지해줘</div>
                <div class="example-item" onclick="setExample('키워드 상태가 어떻게 되나요?')">키워드 상태가 어떻게 되나요?</div>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const message = document.getElementById('message').value;
                const keyword_id = document.getElementById('keyword_id').value;
                const responseDiv = document.getElementById('response');
                
                // 메시지가 비어있으면 알림
                if (!message) {
                    alert('메시지를 입력해주세요.');
                    return;
                }
                
                // 응답 영역에 로딩 표시
                responseDiv.innerHTML = '<strong>처리 중...</strong>';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            keyword_id: keyword_id
                        })
                    });

                    const data = await response.json();
                    
                    if (response.ok) {
                        responseDiv.innerHTML = `<strong>응답:</strong><br>${data.response}`;
                    } else {
                        responseDiv.innerHTML = `<strong>오류:</strong><br>${data.detail || '요청 처리 중 오류가 발생했습니다.'}`;
                    }
                } catch (error) {
                    responseDiv.innerHTML = `<strong>오류:</strong><br>${error.message}`;
                }
            }
            
            function setExample(text) {
                document.getElementById('message').value = text;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"채팅 요청 수신: {request.message}")
        
        # 키워드 ID가 제공된 경우 키워드 정보 가져오기
        keyword_info = None
        if request.keyword_id:
            keyword_info = get_keyword_info(request.keyword_id)
            logger.info(f"키워드 정보 조회 성공: {keyword_info}")
        
        # 사용자 명령 분석
        if keyword_info:
            command_analysis = analyze_user_command(request.message, keyword_info)
            
            # 입찰가 수정 명령인 경우
            if command_analysis["action"] == "update_bid" and "bid_amount" in command_analysis:
                bid_amount = command_analysis["bid_amount"]
                updated_info = update_keyword_bid(request.keyword_id, bid_amount)
                return {"response": f"입찰가가 {bid_amount}원으로 수정되었습니다. 업데이트된 정보: {updated_info}"}
            
            # 상태 변경 명령인 경우
            elif command_analysis["action"] == "update_status" and "status" in command_analysis:
                status = command_analysis["status"]
                updated_info = update_keyword_status(request.keyword_id, status)
                status_text = "활성화" if status else "일시중지"
                return {"response": f"키워드가 {status_text} 상태로 변경되었습니다. 업데이트된 정보: {updated_info}"}
        
        # GPT에 전달할 시스템 프롬프트
        system_prompt = """당신은 네이버 검색광고 키워드 정보를 제공하는 챗봇입니다.
        사용자의 질문에 대해 키워드 정보를 바탕으로 친절하게 답변해주세요.
        
        키워드 ID가 제공되었다면 다음과 같은 작업을 수행할 수 있습니다:
        1. 입찰가 수정 - 예: "입찰가를 700원으로 변경해줘"
        2. 상태 변경 - 예: "키워드를 활성화해줘", "키워드를 일시중지해줘"
        3. 키워드 정보 조회 - 예: "현재 입찰가는 얼마야?"
        
        만약 키워드 ID가 제공되지 않았다면, 사용자에게 키워드 ID를 입력하도록 안내해주세요.
        """
        
        # 키워드 정보가 있는 경우 프롬프트에 추가
        if keyword_info:
            system_prompt += f"\n현재 키워드 정보:\n{keyword_info}"
        
        # GPT API 호출
        logger.info("GPT API 호출 시작")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message}
            ]
        )
        
        logger.info("GPT API 호출 성공")
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 