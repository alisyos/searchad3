from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import time
import logging
import hashlib
import hmac
import base64
import json
import uuid
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
from pydantic import BaseModel

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

# 정적 파일 추가
app.mount("/static", StaticFiles(directory="static"), name="static")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")

# 네이버 검색광고 API 설정
NAVER_API_URL = "https://api.searchad.naver.com"
NAVER_API_KEY = os.getenv("NAVER_API_KEY")
NAVER_API_SECRET = os.getenv("NAVER_API_SECRET")
NAVER_CUSTOMER_ID = os.getenv("NAVER_CUSTOMER_ID")

# Assistant ID
ASSISTANT_ID = "asst_iRibbEvIbbFQq94XlmBevKXu"

# 스레드 저장소 (메모리에 저장, 실제로는 DB에 저장해야 함)
threads_db = {}

if not all([NAVER_API_KEY, NAVER_API_SECRET, NAVER_CUSTOMER_ID]):
    logger.error("네이버 API 관련 환경 변수가 설정되지 않았습니다.")

class ChatRequest(BaseModel):
    message: str
    keyword_id: Optional[str] = None
    thread_id: Optional[str] = None

# API 서명 생성 함수
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

def get_headers(method: str, uri: str):
    """네이버 검색광고 API 호출에 필요한 헤더를 생성합니다."""
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, method, uri, NAVER_API_SECRET)
    return {
        "Content-Type": "application/json",
        "X-Timestamp": timestamp,
        "X-API-KEY": NAVER_API_KEY,
        "X-Customer": NAVER_CUSTOMER_ID,
        "X-Signature": signature
    }

def get_keyword_info(keyword_id: str):
    """네이버 검색광고 API를 통해 키워드 정보를 가져옵니다."""
    try:
        method = "GET"
        uri = f"/ncc/keywords/{keyword_id}"
        base_url = f"{NAVER_API_URL}{uri}"
        headers = get_headers(method, uri)
        
        logger.info(f"키워드 정보 조회 요청: {keyword_id}")
        response = requests.get(base_url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"키워드 정보 조회 실패: status={response.status_code}, response={response.text}")
            
        response.raise_for_status()
        return response.json()[0] if isinstance(response.json(), list) else response.json()
    except Exception as e:
        logger.error(f"키워드 정보 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"키워드 정보를 가져오는데 실패했습니다: {str(e)}")

def update_bid_amount(keyword_id: str, bid_amount: int):
    """네이버 검색광고 API를 통해 키워드 입찰가를 수정합니다."""
    try:
        # 키워드 정보를 먼저 가져와 그룹 ID를 얻습니다
        keyword_info = get_keyword_info(keyword_id)
        
        method = "PUT"
        uri = f"/ncc/keywords/{keyword_id}?fields=bidAmt"
        base_url = f"{NAVER_API_URL}{uri}"
        headers = get_headers(method, uri)
        
        data = {
            "nccKeywordId": keyword_id,
            "nccAdgroupId": keyword_info["nccAdgroupId"],
            "bidAmt": bid_amount
        }
        
        logger.info(f"키워드 입찰가 수정 요청: {keyword_id}, 입찰가: {bid_amount}")
        response = requests.put(base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            logger.error(f"입찰가 수정 실패: status={response.status_code}, response={response.text}")
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"입찰가 수정 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"입찰가 수정에 실패했습니다: {str(e)}")

def update_keyword_status(keyword_id: str, enable: bool):
    """네이버 검색광고 API를 통해 키워드 상태를 활성화/비활성화합니다."""
    try:
        # 키워드 정보를 먼저 가져와 그룹 ID를 얻습니다
        keyword_info = get_keyword_info(keyword_id)
        
        method = "PUT"
        uri = f"/ncc/keywords/{keyword_id}?fields=userLock"
        base_url = f"{NAVER_API_URL}{uri}"
        headers = get_headers(method, uri)
        
        data = {
            "nccKeywordId": keyword_id,
            "nccAdgroupId": keyword_info["nccAdgroupId"],
            "userLock": not enable  # true: 중지, false: 활성화
        }
        
        logger.info(f"키워드 상태 변경 요청: {keyword_id}, 활성화: {enable}")
        response = requests.put(base_url, headers=headers, json=data)
        
        if response.status_code != 200:
            logger.error(f"상태 변경 실패: status={response.status_code}, response={response.text}")
            
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"키워드 상태 변경 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"키워드 상태 변경에 실패했습니다: {str(e)}")

# OpenAI Assistant API 관련 함수
def create_thread():
    """새로운 스레드를 생성합니다."""
    try:
        # 최신 OpenAI API 클라이언트 사용
        client = openai.OpenAI(api_key=openai.api_key)
        thread = client.beta.threads.create()
        return thread.id
    except Exception as e:
        logger.error(f"스레드 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"스레드 생성에 실패했습니다: {str(e)}")

def add_message_to_thread(thread_id, message, keyword_info=None):
    """스레드에 메시지를 추가합니다."""
    try:
        content = message
        if keyword_info:
            content += f"\n\n현재 키워드 정보: {json.dumps(keyword_info, ensure_ascii=False)}"
        
        # 최신 OpenAI API 클라이언트 사용
        client = openai.OpenAI(api_key=openai.api_key)
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        return message
    except Exception as e:
        logger.error(f"메시지 추가 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"메시지 추가에 실패했습니다: {str(e)}")

def run_assistant(thread_id, tools=None):
    """Assistant를 실행하고 응답을 기다립니다."""
    try:
        # 최신 OpenAI API 클라이언트 사용
        client = openai.OpenAI(api_key=openai.api_key)
        
        # Assistant 실행
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID,
            tools=tools
        )
        
        # 실행 완료까지 대기
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:
                raise HTTPException(status_code=500, detail=f"Assistant 실행에 실패했습니다: {run_status.status}")
            
            # 도구 호출이 필요한 경우
            if run_status.status == 'requires_action':
                tool_outputs = []
                for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    result = None
                    # 키워드 정보 조회
                    if function_name == 'get_keyword_info':
                        keyword_id = function_args.get('keyword_id')
                        result = get_keyword_info(keyword_id)
                    # 입찰가 수정
                    elif function_name == 'update_keyword_bid':
                        keyword_id = function_args.get('keyword_id')
                        bid_amount = function_args.get('bid_amount')
                        use_group_bid = function_args.get('use_group_bid', False)
                        result = update_bid_amount(keyword_id, bid_amount)
                    # 상태 변경
                    elif function_name == 'update_keyword_status':
                        keyword_id = function_args.get('keyword_id')
                        enabled = function_args.get('enabled')
                        result = update_keyword_status(keyword_id, enabled)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(result) if result else "작업 실패"
                    })
                
                # 도구 호출 결과 제출
                client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            
            time.sleep(1)  # 잠시 대기
        
        # 모든 메시지 가져오기
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        # 가장 최근 Assistant 메시지 반환
        for message in messages.data:
            if message.role == 'assistant':
                return message.content[0].text.value
        
        return "응답을 받지 못했습니다."
    except Exception as e:
        logger.error(f"Assistant 실행 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assistant 실행에 실패했습니다: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """메인 페이지 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    """Assistants API를 사용한 채팅 처리"""
    try:
        logger.info(f"채팅 요청 수신: {request.message}")
        
        # 스레드 ID 확인 (없으면 새로 생성)
        thread_id = request.thread_id
        if not thread_id or thread_id not in threads_db:
            thread_id = create_thread()
            threads_db[thread_id] = {"created_at": time.time()}
            logger.info(f"새 스레드 생성: {thread_id}")
        
        # 키워드 정보 조회
        keyword_info = None
        if request.keyword_id:
            keyword_info = get_keyword_info(request.keyword_id)
            logger.info(f"키워드 정보 조회 성공: {keyword_info}")
        
        # 도구 정의
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_keyword_info",
                    "description": "네이버 검색광고 API를 통해 키워드 정보를 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword_id": {
                                "type": "string",
                                "description": "조회할 키워드의 ID"
                            }
                        },
                        "required": ["keyword_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_keyword_bid",
                    "description": "네이버 검색광고 API를 통해 키워드의 입찰가를 수정합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword_id": {
                                "type": "string",
                                "description": "수정할 키워드의 ID"
                            },
                            "bid_amount": {
                                "type": "integer",
                                "description": "설정할 입찰가 (70~100000 사이)"
                            },
                            "use_group_bid": {
                                "type": "boolean",
                                "description": "그룹 입찰가 사용 여부"
                            }
                        },
                        "required": ["keyword_id", "bid_amount"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_keyword_status",
                    "description": "네이버 검색광고 API를 통해 키워드의 상태(활성화/일시중지)를 변경합니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword_id": {
                                "type": "string",
                                "description": "상태를 변경할 키워드의 ID"
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "활성화(true) 또는 일시중지(false) 여부"
                            }
                        },
                        "required": ["keyword_id", "enabled"]
                    }
                }
            }
        ]
        
        # 스레드에 메시지 추가
        add_message_to_thread(thread_id, request.message, keyword_info)
        
        # Assistant 실행
        response = run_assistant(thread_id, tools)
        
        return {
            "response": response,
            "thread_id": thread_id
        }
        
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_campaigns():
    """네이버 검색광고 API를 통해 캠페인 목록을 가져옵니다."""
    logger.info("캠페인 목록 가져오기")
    
    method = "GET"
    uri = "/ncc/campaigns"
    base_url = f"{NAVER_API_URL}{uri}"
    headers = get_headers(method, uri)
    
    # API 가이드에 따라 추가적인 쿼리 파라미터 설정
    params = {
        "campaignType": "WEB_SITE",  # 파워링크 유형
        "recordSize": 1000  # 최대 1000개 조회
    }
    
    try:
        logger.info(f"캠페인 조회 요청: params={params}, headers={headers}")
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code != 200:
            logger.error(f"캠페인 조회 실패: status={response.status_code}, response={response.text}")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"캠페인 정보 요청 실패: {e}")
        raise HTTPException(status_code=500, detail=f"캠페인 정보를 가져오는데 실패했습니다. {str(e)}")

def get_adgroups(campaign_id: str):
    """네이버 검색광고 API를 통해 특정 캠페인의 그룹 목록을 가져옵니다."""
    logger.info(f"그룹 목록 가져오기: campaign_id={campaign_id}")
    
    method = "GET"
    uri = "/ncc/adgroups"
    base_url = f"{NAVER_API_URL}{uri}"
    headers = get_headers(method, uri)
    
    # 쿼리 파라미터 설정
    params = {
        "nccCampaignId": campaign_id
    }
    
    try:
        logger.info(f"그룹 조회 요청: params={params}")
        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code != 200:
            logger.error(f"그룹 조회 실패: status={response.status_code}, response={response.text}")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"그룹 정보 요청 실패: {e}")
        raise HTTPException(status_code=500, detail=f"그룹 정보를 가져오는데 실패했습니다. {str(e)}")

def get_keywords(adgroup_id: str):
    """네이버 검색광고 API를 통해 키워드 정보를 가져옵니다."""
    logger.info(f"키워드 정보 가져오기: adgroup_id={adgroup_id}")
    
    method = "GET"
    
    try:
        # 키워드 ID로 검색하는 경우와 그룹 ID로 검색하는 경우 구분
        if adgroup_id.startswith("nkw-"):
            # 키워드 ID로 검색
            keyword_id = adgroup_id
            uri = f"/ncc/keywords/{keyword_id}"
            base_url = f"{NAVER_API_URL}{uri}"
            headers = get_headers(method, uri)
            
            logger.info(f"키워드 상세 정보 조회 요청: keyword_id={keyword_id}")
            response = requests.get(base_url, headers=headers)
        else:
            # 그룹 ID로 검색
            uri = "/ncc/keywords"
            base_url = f"{NAVER_API_URL}{uri}"
            headers = get_headers(method, uri)
            
            params = {
                "nccAdgroupId": adgroup_id
            }
            
            logger.info(f"그룹의 키워드 목록 조회 요청: adgroup_id={adgroup_id}")
            response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code != 200:
            logger.error(f"키워드 조회 실패: status={response.status_code}, response={response.text}")
            
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"키워드 정보 요청 실패: {e}")
        raise HTTPException(status_code=500, detail=f"키워드 정보를 가져오는데 실패했습니다. {str(e)}")

@app.get("/api/campaigns")
def api_campaigns():
    """모든 캠페인 목록을 반환합니다."""
    try:
        campaigns = get_campaigns()
        return {"campaigns": campaigns}
    except Exception as e:
        logger.error(f"캠페인 API 호출 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adgroups/{campaign_id}")
def api_adgroups(campaign_id: str):
    """특정 캠페인의 그룹 목록을 반환합니다."""
    try:
        adgroups = get_adgroups(campaign_id)
        return {"adgroups": adgroups}
    except Exception as e:
        logger.error(f"그룹 API 호출 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/keywords/{id}")
def get_keywords_endpoint(id: str):
    """키워드 정보를 반환합니다."""
    try:
        keywords = get_keywords(id)
        return {"keywords": keywords}
    except Exception as e:
        logger.error(f"키워드 정보 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Render.com에서 제공하는 PORT 환경 변수 사용
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port) 