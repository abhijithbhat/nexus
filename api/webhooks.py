import time
import asyncio
from urllib.parse import parse_qsl
from fastapi import APIRouter, Request, Header, HTTPException, BackgroundTasks
from fastapi.responses import Response
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Rate limiting memory storage
_rate_limits = {}
_rate_limit_lock = asyncio.Lock()

async def process_and_reply(orchestrator, whatsapp, body: str, from_number: str):
    try:
        response = await orchestrator.process_message(body, from_number)
        if response and response != "Unauthorized":
            whatsapp.send_message(from_number, response)
    except Exception as e:
        logger.error(f"Error processing webhook message in background task: {e}")

@router.post("/webhook/whatsapp")
async def webhook_whatsapp(
    request: Request,
    background_tasks: BackgroundTasks,
    x_twilio_signature: str = Header(None)
):
    form_data_bytes = await request.body()
    form_data_str = form_data_bytes.decode("utf-8")
    
    # Parse form parameters
    form_params = dict(parse_qsl(form_data_str))
    
    whatsapp = request.app.state.whatsapp
    url = str(request.url)
    
    # Validate request signature
    if not whatsapp.validate_request(url, form_params, x_twilio_signature):
        logger.warning(f"Twilio signature validation failed. Signature: {x_twilio_signature}, URL: {url}")
        raise HTTPException(status_code=403, detail="Forbidden")
        
    parsed = whatsapp.parse_incoming_webhook(form_params)
    from_number = parsed["from_number"]
    body = parsed["body"]
    
    if from_number != settings.user_whatsapp_number:
        logger.warning(f"Blocked unauthorized webhook source number: {from_number}")
        raise HTTPException(status_code=403, detail="Forbidden")
        
    # Rate limit: max 1 message per 2 seconds per number
    async with _rate_limit_lock:
        now = time.time()
        last_time = _rate_limits.get(from_number, 0)
        if now - last_time < 2.0:
            logger.warning(f"Rate limit exceeded for {from_number}. Dropping message.")
            raise HTTPException(status_code=429, detail="Too Many Requests")
        _rate_limits[from_number] = now
        
    # Execute processing async
    background_tasks.add_task(
        process_and_reply,
        request.app.state.orchestrator,
        whatsapp,
        body,
        from_number
    )
    
    twiml_response = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
    return Response(content=twiml_response, media_type="application/xml")
