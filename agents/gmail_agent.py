from connectors.gmail import GmailConnector
from utils.llm_factory import get_primary_client
from utils.logger import get_logger

logger = get_logger(__name__)


class GmailAgent:
    def __init__(self):
        self.gmail = GmailConnector()
        self.llm = get_primary_client()
    
    async def run(self, task: str, context: str) -> str:
        task_lower = task.lower()
        
        if ("send" in task_lower or "draft" in task_lower or "write" in task_lower) and "email" in task_lower:
            return await self._handle_send(task, context)
        elif "check" in task_lower or "unread" in task_lower or "inbox" in task_lower:
            return await self._handle_check(task, context)
        else:
            return await self._handle_check(task, context)
    
    async def _handle_check(self, task: str, context: str) -> str:
        if not self.gmail.is_available:
            return "📭 Gmail is not configured. Set up Google OAuth credentials to enable email access."
        
        emails = self.gmail.get_unread_emails(max_results=5)
        if not emails:
            return "📭 No unread emails right now."
        
        system_prompt = "You are NEXUS summarizing emails for the user. Be concise and highlight what needs action."
        user_message = f"Task: {task}\nEmails:\n{emails}\nUser context: {context[:200]}"
        
        try:
            summary = await self.llm.generate(system_prompt, user_message, temperature=0.3)
            return f"📧 **Email Summary**\n\n{summary}"
        except Exception as e:
            # Fallback: format manually
            logger.error(f"Email summary generation failed: {e}")
            lines = [f"📧 {len(emails)} unread emails:"]
            for e_item in emails:
                lines.append(
                    f"• From: {e_item['from'][:40]}\n"
                    f"  Subject: {e_item['subject'][:60]}\n"
                    f"  {e_item['snippet'][:100]}"
                )
            return "\n\n".join(lines)
    
    async def _handle_send(self, task: str, context: str) -> str:
        if not self.gmail.is_available:
            return "❌ Gmail is not configured. Set up Google OAuth credentials to enable sending emails."
        
        # Extract recipient, subject, body from task via Gemini
        system_prompt = "Extract email details from the user's request. Return JSON only."
        user_message = (
            f"Task: {task}\n"
            f"Context: {context[:300]}\n\n"
            f"Extract: {{\"to\": \"email@example.com\", \"subject\": \"...\", \"body\": \"...\"}}"
        )
        
        try:
            details = await self.llm.generate_json(system_prompt, user_message)
            to = details.get("to", "")
            subject = details.get("subject", "")
            body = details.get("body", "")
            
            if not to or "@" not in to:
                return "❌ Could not determine recipient email. Please specify the email address."
            
            success = self.gmail.send_email(to, subject, body)
            if success:
                return f"✅ Email sent to {to}\nSubject: {subject}"
            else:
                return "❌ Failed to send email. Check Gmail connector configuration."
        except Exception as e:
            logger.error(f"Gmail send error: {e}")
            return f"Error sending email: {e}"
