import json
from datetime import datetime
from typing import TypedDict, Optional
import pytz
from langgraph.graph import StateGraph, END
from utils.config import settings
from utils.logger import get_logger
from utils.gemini_client import GeminiClient
from memory.memory_manager import MemoryManager

logger = get_logger(__name__)

class NexusState(TypedDict):
    user_input: str
    from_number: str
    recalled_memory: str
    user_context: str
    routing_plan: str
    active_agent: str
    agent_result: str
    final_response: str
    action_taken: bool
    error: Optional[str]

NEXUS_SYSTEM_PROMPT = """
You are NEXUS, a proactive personal AI agent built specifically for {user_name}.

ABOUT {user_name}:
• Interests: {user_interests}
• Current goals: {user_goals}
• Location: {user_location}
• You know them deeply through persistent memory built over time.

WHAT YOU KNOW RIGHT NOW (from memory):
{user_context}

RELEVANT MEMORY FOR THIS CONVERSATION:
{recalled_context}

CURRENT DATE & TIME: {datetime} (IST)

YOUR BEHAVIORAL RULES:
1. You are PROACTIVE. Always think: what does this person actually need, not just what did they literally ask?
2. You have PERSISTENT MEMORY. Never pretend you don't know something that's in your context above.
3. Be CONCISE for quick questions. Be DETAILED for complex requests. Never pad responses.
4. When you take or plan a real-world action (send email, schedule event, write code), state it clearly.
5. You have access to sub-agents: researcher (web search + synthesis), coder (writes + runs Python), scheduler (manages events/reminders), communicator (drafts messages). Use them when appropriate.
6. Speak like a brilliant friend who knows this person well — not like a corporate assistant.
7. If uncertain, say so. NEXUS never fabricates.
8. You may initiate things. If memory tells you a deadline is tomorrow, bring it up unprompted.
"""

class NexusOrchestrator:
    def __init__(self, memory_manager: MemoryManager, gemini_client: GeminiClient):
        self.memory_manager = memory_manager
        self.gemini_client = gemini_client
        
        # Deferred import to prevent circular dependency issues
        from agents.researcher import ResearcherAgent
        from agents.coder import CoderAgent
        from agents.scheduler import SchedulerAgent
        from agents.communicator import CommunicatorAgent
        
        self.researcher_agent = ResearcherAgent()
        self.coder_agent = CoderAgent()
        self.scheduler_agent = SchedulerAgent(memory_manager)
        self.communicator_agent = CommunicatorAgent()
        
        # Build and compile the workflow graph
        self.graph = self._build_graph()
        logger.info("LangGraph orchestrator initialized and compiled.")

    async def recall_memory_node(self, state: NexusState) -> dict:
        recalled = await self.memory_manager.recall(state["user_input"])
        return {"recalled_memory": recalled}

    async def build_context_node(self, state: NexusState) -> dict:
        context = await self.memory_manager.get_full_user_context()
        return {"user_context": context}

    async def orchestrate_node(self, state: NexusState) -> dict:
        logger.info(f"Routing request: '{state['user_input'][:60]}'")
        
        system_prompt = (
            "You are the routing intelligence of NEXUS. Decide which specialized agent to use based on the user's message and context. Be precise."
        )
        
        user_message = (
            f"User message: {state['user_input']}\n\n"
            f"User context summary: {state['user_context'][:500]}\n\n"
            f"Relevant memory: {state['recalled_memory'][:300]}\n\n"
            f"AVAILABLE AGENTS:\n"
            f"- researcher: Use when user wants information researched from the web. Triggers: 'research', 'find out', 'what is', 'latest on', 'summarize', 'look up', 'tell me about'\n"
            f"- coder: Use when user wants code written, a script built, or a technical solution generated. Triggers: 'write a script', 'build', 'create a function', 'code that does', 'automate'\n"
            f"- scheduler: Use when user wants reminders, scheduling, or event management. Triggers: 'remind me', 'schedule', 'add to calendar', 'set alarm', 'deadline'\n"
            f"- communicator: Use when user wants to draft a message, email, or document. Triggers: 'draft', 'write an email', 'compose', 'prepare a message'\n"
            f"- direct: Use for greetings, simple factual questions, opinions, casual conversation, anything that does not need web search or code.\n\n"
            f"Output ONLY valid JSON:\n"
            f"{{\n"
            f"  \"agent\": \"researcher|coder|scheduler|communicator|direct\",\n"
            f"  \"task\": \"specific task description for the agent\",\n"
            f"  \"reason\": \"one sentence\"\n"
            f"}}"
        )
        
        try:
            plan = await self.gemini_client.generate_json(system_prompt, user_message, temperature=0.1)
            agent = plan.get("agent", "direct")
            if agent not in ["researcher", "coder", "scheduler", "communicator", "direct"]:
                agent = "direct"
                
            logger.info(f"Routed to '{agent}' agent. Reason: {plan.get('reason')}")
            return {
                "routing_plan": json.dumps(plan),
                "active_agent": agent
            }
        except Exception as e:
            logger.error(f"Routing failed: {e}. Defaulting to direct_response.")
            fallback_plan = {"agent": "direct", "task": state["user_input"], "reason": f"Routing error: {e}"}
            return {
                "routing_plan": json.dumps(fallback_plan),
                "active_agent": "direct"
            }

    async def researcher_node(self, state: NexusState) -> dict:
        plan = json.loads(state["routing_plan"])
        task = plan.get("task", state["user_input"])
        result = await self.researcher_agent.run(task, state["user_context"])
        return {"agent_result": result, "action_taken": False}

    async def coder_node(self, state: NexusState) -> dict:
        plan = json.loads(state["routing_plan"])
        task = plan.get("task", state["user_input"])
        result = await self.coder_agent.run(task, state["user_context"])
        return {"agent_result": result, "action_taken": False}

    async def scheduler_node(self, state: NexusState) -> dict:
        plan = json.loads(state["routing_plan"])
        task = plan.get("task", state["user_input"])
        result = await self.scheduler_agent.run(task, state["user_context"])
        return {"agent_result": result, "action_taken": True}

    async def communicator_node(self, state: NexusState) -> dict:
        plan = json.loads(state["routing_plan"])
        task = plan.get("task", state["user_input"])
        result = await self.communicator_agent.run(task, state["user_context"])
        return {"agent_result": result, "action_taken": False}

    async def direct_response_node(self, state: NexusState) -> dict:
        tz = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        
        filled_prompt = NEXUS_SYSTEM_PROMPT.format(
            user_name=settings.user_name,
            user_interests=settings.user_interests,
            user_goals=settings.user_goals,
            user_location=settings.user_location,
            user_context=state["user_context"],
            recalled_context=state["recalled_memory"],
            datetime=now_ist
        )
        
        response = await self.gemini_client.generate(
            system_prompt=filled_prompt,
            user_message=state["user_input"]
        )
        return {"agent_result": response, "action_taken": False}

    async def store_memory_node(self, state: NexusState) -> dict:
        # Save raw conversation inputs
        await self.memory_manager.remember(
            text=f"User said: {state['user_input']}",
            type="conversation",
            source="whatsapp",
            importance=0.4
        )
        
        agent = state["active_agent"]
        result = state["agent_result"]
        
        # Save agent results if they are not direct conversation
        if result and agent != "direct":
            await self.memory_manager.remember(
                text=f"Agent ({agent}) result: {result}",
                type=f"{agent}_result",
                source="orchestrator",
                importance=0.3
            )
        return {"action_taken": state.get("action_taken", False)}

    async def format_response_node(self, state: NexusState) -> dict:
        system_prompt = (
            f"Format the following content as a WhatsApp message from NEXUS to {settings.user_name}. Rules:\n"
            "- Maximum 1500 characters total\n"
            "- Use line breaks generously for readability\n"
            "- Use emoji only where it adds real meaning (not decoration)\n"
            "- If content is longer than 1500 chars, prioritize the most important information and end with '...(more available on request)'\n"
            "- Never start with 'I' — start with the substance\n"
            "- No corporate-speak. Natural, warm, precise."
        )
        
        user_message = (
            f"Agent used: {state['active_agent']}\n"
            f"Raw content to format:\n{state['agent_result']}"
        )
        
        try:
            formatted = await self.gemini_client.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.3
            )
            return {"final_response": formatted}
        except Exception as e:
            logger.error(f"Response formatting failed: {e}. Using raw text.")
            return {"final_response": state["agent_result"][:1500]}

    def _build_graph(self):
        graph = StateGraph(NexusState)

        # Define graph nodes
        graph.add_node("recall_memory", self.recall_memory_node)
        graph.add_node("build_context", self.build_context_node)
        graph.add_node("orchestrate", self.orchestrate_node)
        graph.add_node("researcher", self.researcher_node)
        graph.add_node("coder", self.coder_node)
        graph.add_node("scheduler", self.scheduler_node)
        graph.add_node("communicator", self.communicator_node)
        graph.add_node("direct_response", self.direct_response_node)
        graph.add_node("store_memory", self.store_memory_node)
        graph.add_node("format_response", self.format_response_node)

        # Define edge routing
        graph.set_entry_point("recall_memory")
        graph.add_edge("recall_memory", "build_context")
        graph.add_edge("build_context", "orchestrate")

        # Routing decision edge
        graph.add_conditional_edges(
            "orchestrate",
            lambda state: state["active_agent"],
            {
                "researcher": "researcher",
                "coder": "coder",
                "scheduler": "scheduler",
                "communicator": "communicator",
                "direct": "direct_response"
            }
        )

        # Connect all specialist agents back to memory storage
        for agent_node in ["researcher", "coder", "scheduler", "communicator", "direct_response"]:
            graph.add_edge(agent_node, "store_memory")

        graph.add_edge("store_memory", "format_response")
        graph.add_edge("format_response", END)

        return graph.compile()

    async def process_message(self, user_input: str, from_number: str) -> str:
        if from_number != settings.user_whatsapp_number:
            logger.warning(f"Blocked message from unauthorized number: {from_number}")
            return "Unauthorized"
            
        initial_state: NexusState = {
            "user_input": user_input,
            "from_number": from_number,
            "recalled_memory": "",
            "user_context": "",
            "routing_plan": "",
            "active_agent": "",
            "agent_result": "",
            "final_response": "",
            "action_taken": False,
            "error": None
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state["final_response"]
        except Exception as e:
            logger.error(f"Error executing LangGraph orchestrator state machine: {e}")
            return f"My systems encountered an error while processing your request: {e}"
