"""Twilio WhatsApp inbound webhook — the main entry point for user messages."""
import asyncio
import time
from fastapi import APIRouter, Request, Response
from utils.config import settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Simple in-memory rate limiter: {phone_number: last_processed_timestamp}
_last_msg_time: dict[str, float] = {}
_RATE_LIMIT_SECONDS = 2.0
_lock = asyncio.Lock()


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> Response:
    """Receive a WhatsApp message from Twilio and process it asynchronously."""
    form = await request.form()
    form_data = dict(form)

    # 1. Validate Twilio signature (skip in dev for easier testing)
    wa = request.app.state.whatsapp
    if settings.app_env == "production":
        signature = request.headers.get("X-Twilio-Signature", "")
        url = str(request.url)
        if not wa.validate_request(url, form_data, signature):
            logger.warning("[Webhook] Invalid Twilio signature — rejecting")
            return Response(status_code=403)

    # 2. Parse the message
    parsed = wa.parse_incoming_webhook(form_data)
    body = parsed["body"]
    sender = parsed["from_number"]

    if not body:
        return Response(content="<Response></Response>", media_type="application/xml")

    # 3. Rate limit: ignore if last message from this sender was < 2s ago
    async with _lock:
        now = time.time()
        last = _last_msg_time.get(sender, 0)
        if now - last < _RATE_LIMIT_SECONDS:
            logger.debug(f"[Webhook] Rate limited {sender}")
            return Response(content="<Response></Response>", media_type="application/xml")
        _last_msg_time[sender] = now

    logger.info(f"[Webhook] Message from {sender}: {body[:80]}...")

    # 4. Process asynchronously — don't block the webhook response
    asyncio.create_task(_process_message(request, body, sender))

    # 5. Return empty TwiML (Twilio expects this)
    return Response(content="<Response></Response>", media_type="application/xml")


async def _process_message(request: Request, body: str, sender: str) -> None:
    """Process the user message through the orchestrator and send a reply."""
    try:
        # Store the conversation in memory
        memory = request.app.state.memory_manager
        await memory.remember(
            f"User said: {body}",
            type="conversation",
            source="whatsapp",
            importance=0.6,
        )

        # Run through the orchestrator
        orchestrator = request.app.state.orchestrator
        response_text = await orchestrator.process(body)

        # Store the response in memory
        await memory.remember(
            f"NEXUS replied: {response_text[:300]}",
            type="conversation",
            source="nexus_response",
            importance=0.4,
        )

        # Send reply via WhatsApp
        wa = request.app.state.whatsapp
        wa.send_message(sender, response_text)

    except Exception as exc:
        logger.error(f"[Webhook] Processing failed: {exc}", exc_info=True)
        try:
            wa = request.app.state.whatsapp
            wa.send_message(sender, f"⚠️ NEXUS encountered an error: {str(exc)[:200]}")
        except Exception:
            pass
