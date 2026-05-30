"""Core orchestrator — LangGraph StateGraph brain that routes messages to agents."""
import asyncio
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from utils.gemini_client import gemini_client
from utils.config import settings
from utils.logger import get_logger
from memory.memory_manager import MemoryManager
from agents.researcher import ResearcherAgent
from agents.coder import CoderAgent
from agents.scheduler import SchedulerAgent
from agents.communicator import CommunicatorAgent

logger = get_logger(__name__)


# ── State definition ──────────────────────────────────────────────────
class NexusState(TypedDict):
    user_message: str
    intent: str
    memory_context: str
    user_profile: str
    agent_outputs: dict
    final_response: str
    needs_followup: bool
    error: str


# ── System prompt for intent classification ───────────────────────────
INTENT_SYSTEM_PROMPT = """You are NEXUS, a personal intelligence system for {user_name}.
Classify the user's message into exactly one intent.

Available intents:
- "research" — needs web search, information lookup, topic exploration
- "code" — programming help, debugging, code generation, technical explanation
- "schedule" — mentions dates, deadlines, reminders, events, meetings
- "communicate" — draft a message, email, or communication
- "casual" — casual chat, greeting, small talk, status check
- "memory" — asks about something NEXUS should remember or recall
- "status" — asks about NEXUS itself, what it's been doing, system status

Return ONLY JSON: {{"intent": "...", "confidence": 0.XX, "reasoning": "one sentence"}}
"""

# ── System prompt for final response generation ──────────────────────
RESPONSE_SYSTEM_PROMPT = """You are NEXUS, {user_name}'s proactive personal intelligence assistant.
You are conversational, helpful, and proactive. You remember past conversations and adapt your tone.

Key traits:
- Warm but concise (WhatsApp messages should be scannable)
- Emoji usage: moderate, purposeful
- Proactive: suggest next steps, mention relevant upcoming events
- Personal: reference what you know about the user
- If you used an agent's output, integrate it naturally — don't say "the research agent found..."

USER CONTEXT:
{user_context}

MEMORY CONTEXT:
{memory_context}
"""


class Orchestrator:
    """LangGraph-powered message processing pipeline."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        researcher: ResearcherAgent,
        coder: CoderAgent,
        scheduler: SchedulerAgent,
        communicator: CommunicatorAgent,
    ) -> None:
        self._memory = memory_manager
        self._researcher = researcher
        self._coder = coder
        self._scheduler = scheduler
        self._communicator = communicator
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph processing pipeline."""
        graph = StateGraph(NexusState)

        # Add nodes
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("gather_context", self._gather_context)
        graph.add_node("route_to_agent", self._route_to_agent)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("extract_events", self._extract_events)

        # Define edges
        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "gather_context")
        graph.add_edge("gather_context", "route_to_agent")
        graph.add_edge("route_to_agent", "extract_events")
        graph.add_edge("extract_events", "generate_response")
        graph.add_edge("generate_response", END)

        return graph.compile()

    # ── Graph nodes ───────────────────────────────────────────────────

    async def _classify_intent(self, state: NexusState) -> dict:
        """Classify the user's message intent."""
        try:
            system = INTENT_SYSTEM_PROMPT.format(user_name=settings.user_name)
            result = await gemini_client.generate_json(system, state["user_message"])
            intent = result.get("intent", "casual")
            logger.info(f"[Orchestrator] Intent: {intent} (confidence: {result.get('confidence', '?')})")
            return {"intent": intent}
        except Exception as exc:
            logger.error(f"[Orchestrator] Intent classification failed: {exc}")
            return {"intent": "casual", "error": str(exc)}

    async def _gather_context(self, state: NexusState) -> dict:
        """Gather relevant memory and user context."""
        try:
            memory_context = await self._memory.recall(state["user_message"])
            user_profile = await self._memory.get_full_user_context()
            return {"memory_context": memory_context, "user_profile": user_profile}
        except Exception as exc:
            logger.error(f"[Orchestrator] Context gathering failed: {exc}")
            return {"memory_context": "No memory available", "user_profile": ""}

    async def _route_to_agent(self, state: NexusState) -> dict:
        """Route the message to the appropriate specialized agent."""
        intent = state["intent"]
        msg = state["user_message"]
        ctx = state.get("user_profile", "")
        outputs = {}

        try:
            if intent == "research":
                outputs["research"] = await self._researcher.research(msg, user_context=ctx)
            elif intent == "code":
                outputs["code"] = await self._coder.generate_code(msg, user_context=ctx)
            elif intent == "communicate":
                outputs["communication"] = await self._communicator.draft_message(msg, user_context=ctx)
            elif intent == "memory":
                outputs["memory"] = state.get("memory_context", "No relevant memories found.")
            elif intent == "status":
                outputs["status"] = f"NEXUS is running. Memory entries: {self._memory._vs.count()}"
            # casual and schedule don't need specific agent routing
        except Exception as exc:
            logger.error(f"[Orchestrator] Agent routing failed: {exc}")
            outputs["error"] = str(exc)

        return {"agent_outputs": outputs}

    async def _extract_events(self, state: NexusState) -> dict:
        """Check if the message contains any events/deadlines to store."""
        try:
            events = await self._scheduler.extract_and_store_events(state["user_message"])
            if events:
                state["agent_outputs"]["stored_events"] = events
                logger.info(f"[Orchestrator] Stored {len(events)} events")
        except Exception as exc:
            logger.debug(f"[Orchestrator] Event extraction skipped: {exc}")
        return {"agent_outputs": state.get("agent_outputs", {})}

    async def _generate_response(self, state: NexusState) -> dict:
        """Generate the final response combining all context and agent outputs."""
        try:
            agent_context = ""
            for key, value in state.get("agent_outputs", {}).items():
                if isinstance(value, str):
                    agent_context += f"\n[{key} result]: {value[:800]}\n"
                elif isinstance(value, list):
                    agent_context += f"\n[{key}]: {str(value)[:400]}\n"

            system = RESPONSE_SYSTEM_PROMPT.format(
                user_name=settings.user_name,
                user_context=state.get("user_profile", ""),
                memory_context=state.get("memory_context", ""),
            )

            prompt = (
                f"User message: {state['user_message']}\n"
                f"Detected intent: {state['intent']}\n"
                f"Agent results:\n{agent_context}\n\n"
                "Generate a natural, helpful WhatsApp response. "
                "Keep it concise (under 300 words for WhatsApp)."
            )

            response = await gemini_client.generate(system, prompt, temperature=0.7)
            return {"final_response": response}
        except Exception as exc:
            logger.error(f"[Orchestrator] Response generation failed: {exc}")
            return {"final_response": f"I'm having trouble processing that right now. Error: {str(exc)[:100]}"}

    # ── Public interface ──────────────────────────────────────────────

    async def process(self, user_message: str) -> str:
        """Process a user message through the full pipeline. Returns response text."""
        logger.info(f"[Orchestrator] Processing: {user_message[:80]}...")

        initial_state: NexusState = {
            "user_message": user_message,
            "intent": "",
            "memory_context": "",
            "user_profile": "",
            "agent_outputs": {},
            "final_response": "",
            "needs_followup": False,
            "error": "",
        }

        try:
            result = await self._graph.ainvoke(initial_state)
            response = result.get("final_response", "I couldn't generate a response.")
            logger.info(f"[Orchestrator] Response: {response[:100]}...")
            return response
        except Exception as exc:
            logger.error(f"[Orchestrator] Pipeline failed: {exc}", exc_info=True)
            return f"⚠️ NEXUS pipeline error: {str(exc)[:200]}"
