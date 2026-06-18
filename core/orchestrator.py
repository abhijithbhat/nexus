import json
from datetime import datetime
from typing import TypedDict, Optional
import pytz
from langgraph.graph import StateGraph, END
from utils.config import settings
from utils.logger import get_logger
from utils.gemini_client import GeminiClient
from utils.groq_client import GroqClient
from memory.memory_manager import MemoryManager
from core.session_manager import SessionManager
from core.planner import TaskPlanner
from core.goal_tracker import GoalTracker

logger = get_logger(__name__)


class NexusState(TypedDict):
    user_input: str
    from_number: str
    session_history: str
    recalled_memory: str
    user_context: str
    task_plan: str
    plan_step_results: list
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

SESSION CONVERSATION HISTORY:
{session_history}

CURRENT DATE & TIME: {datetime} (IST)

YOUR BEHAVIORAL RULES:
1. You are PROACTIVE. Always think: what does this person actually need, not just what did they literally ask?
2. You have PERSISTENT MEMORY. Never pretend you don't know something that's in your context above.
3. Be CONCISE for quick questions. Be DETAILED for complex requests. Never pad responses.
4. When you take or plan a real-world action (send email, schedule event, write code), state it clearly.
5. You have access to sub-agents: researcher (web search + synthesis), coder (writes + runs Python), scheduler (manages events/reminders), communicator (drafts messages), gmail (email read/send). Use them when appropriate.
6. Speak like a brilliant friend who knows this person well — not like a corporate assistant.
7. If uncertain, say so. NEXUS never fabricates.
8. You may initiate things. If memory tells you a deadline is tomorrow, bring it up unprompted.
9. Use the session conversation history above to maintain continuity. Don't repeat yourself.
"""


class NexusOrchestrator:
    def __init__(self, memory_manager: MemoryManager, gemini_client: GeminiClient):
        self.memory_manager = memory_manager
        self.gemini_client = gemini_client
        self.groq_client = GroqClient()
        self.session_manager = SessionManager()
        
        # Use Groq for utility calls (planner, goal tracker) when available
        utility_client = self.groq_client if self.groq_client.is_available else gemini_client
        self.planner = TaskPlanner(utility_client)
        self.goal_tracker = GoalTracker(
            memory_manager.knowledge_graph, utility_client, settings.user_goals
        )
        
        # Deferred import to prevent circular dependency issues
        from agents.researcher import ResearcherAgent
        from agents.coder import CoderAgent
        from agents.scheduler import SchedulerAgent
        from agents.communicator import CommunicatorAgent
        from agents.gmail_agent import GmailAgent
        
        self.researcher_agent = ResearcherAgent()
        self.coder_agent = CoderAgent()
        self.scheduler_agent = SchedulerAgent(memory_manager)
        self.communicator_agent = CommunicatorAgent()
        self.gmail_agent = GmailAgent()
        
        # Build and compile the workflow graph
        self.graph = self._build_graph()
        logger.info(f"LangGraph orchestrator initialized. Groq available: {self.groq_client.is_available}")

    async def load_session_node(self, state: NexusState) -> dict:
        history = await self.session_manager.get_formatted_history(state["from_number"])
        return {"session_history": history}

    async def recall_memory_node(self, state: NexusState) -> dict:
        recalled = await self.memory_manager.recall(state["user_input"])
        return {"recalled_memory": recalled}

    async def build_context_node(self, state: NexusState) -> dict:
        context = await self.memory_manager.get_full_user_context()
        return {"user_context": context}

    async def plan_task_node(self, state: NexusState) -> dict:
        """Local keyword-based complexity check. No Gemini call needed."""
        text = state["user_input"].lower()
        
        # Only flag as complex if message explicitly chains multiple tasks
        chain_words = [" then ", " and then ", " after that ", " followed by "]
        is_complex = any(w in text for w in chain_words) and len(text) > 60
        
        if is_complex:
            # Only call Gemini planner for genuinely complex multi-step requests
            plan = await self.planner.analyze(state["user_input"], state["user_context"])
            return {"task_plan": json.dumps(plan), "plan_step_results": []}
        
        return {"task_plan": "{}", "plan_step_results": []}

    # ── Keyword-based routing (no Gemini call needed) ─────────────────
    ROUTE_KEYWORDS = {
        "researcher": [
            "research", "find out", "what is", "latest on", "summarize",
            "look up", "tell me about", "search for", "who is", "explain",
            "how does", "what are", "news about", "trending"
        ],
        "coder": [
            "write a script", "write code", "build a", "create a function",
            "code that", "automate", "python", "program", "algorithm",
            "debug", "fix this code", "implement"
        ],
        "scheduler": [
            "remind me", "schedule", "add to calendar", "set alarm",
            "deadline", "reminder", "event on", "don't forget"
        ],
        "communicator": [
            "draft", "write an email", "compose", "prepare a message",
            "write a letter", "write a post"
        ],
        "gmail": [
            "check email", "check my email", "inbox", "unread email",
            "send email", "send an email", "reply to email", "my emails"
        ],
    }

    def _route_locally(self, text: str) -> str:
        """Fast keyword-based routing. Returns agent name or 'direct'."""
        text_lower = text.lower()
        for agent, keywords in self.ROUTE_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return agent
        return "direct"

    async def orchestrate_node(self, state: NexusState) -> dict:
        logger.info(f"Routing request: '{state['user_input'][:60]}'")
        
        # Check if task planner identified a multi-step plan
        task_plan = json.loads(state.get("task_plan", "{}"))
        if task_plan.get("is_complex", False) and task_plan.get("steps"):
            return await self._execute_multi_step_plan(state, task_plan)
        
        # Try smart LLM routing via Groq (fast + free), fall back to keywords
        if self.groq_client.is_available:
            agent, task = await self._route_with_groq(state)
        else:
            agent = self._route_locally(state["user_input"])
            task = state["user_input"]
        
        logger.info(f"Routed to '{agent}' agent.")
        plan = {"agent": agent, "task": task, "reason": "groq" if self.groq_client.is_available else "keyword"}
        return {
            "routing_plan": json.dumps(plan),
            "active_agent": agent
        }
    
    async def _route_with_groq(self, state: NexusState) -> tuple[str, str]:
        """Use Groq (Llama 3.3) for intelligent routing — fast and free."""
        system_prompt = (
            "You are the routing intelligence of NEXUS. Pick the right agent for the user's message. "
            "Output ONLY valid JSON with 'agent' and 'task' keys."
        )
        user_message = (
            f"User message: {state['user_input']}\n"
            f"Context: {state['user_context'][:300]}\n\n"
            f"AGENTS:\n"
            f"- researcher: web research, information lookup\n"
            f"- coder: write/run code, scripts, technical solutions\n"
            f"- scheduler: reminders, scheduling, calendar events\n"
            f"- communicator: draft messages, emails, posts\n"
            f"- gmail: check inbox, send/read emails\n"
            f"- direct: greetings, opinions, casual chat, simple questions\n\n"
            f'{{"agent": "one of the above", "task": "specific task for agent"}}'
        )
        try:
            result = await self.groq_client.generate_json(system_prompt, user_message, temperature=0.1)
            agent = result.get("agent", "direct")
            valid = ["researcher", "coder", "scheduler", "communicator", "gmail", "direct"]
            if agent not in valid:
                agent = "direct"
            task = result.get("task", state["user_input"])
            return agent, task
        except Exception as e:
            logger.warning(f"Groq routing failed: {e}. Falling back to keywords.")
            return self._route_locally(state["user_input"]), state["user_input"]

    async def _execute_multi_step_plan(self, state: NexusState, task_plan: dict) -> dict:
        """Execute multi-step plan sequentially, collecting results."""
        steps = task_plan.get("steps", [])
        step_results = []
        accumulated_context = state["user_context"]
        
        agent_map = {
            "researcher": self.researcher_agent,
            "coder": self.coder_agent,
            "scheduler": self.scheduler_agent,
            "communicator": self.communicator_agent,
            "gmail": self.gmail_agent,
        }
        
        for step in steps:
            agent_name = step.get("agent", "direct")
            task = step.get("task", "")
            step_num = step.get("step", 0)
            
            # Add previous step results as context
            if step_results:
                prev_context = "\n".join([
                    f"Step {i+1} result: {r[:300]}" for i, r in enumerate(step_results)
                ])
                task = f"{task}\n\nContext from previous steps:\n{prev_context}"
            
            if agent_name in agent_map:
                try:
                    result = await agent_map[agent_name].run(task, accumulated_context)
                    step_results.append(result)
                    logger.info(f"Multi-step plan: Step {step_num} ({agent_name}) completed.")
                except Exception as e:
                    step_results.append(f"Error in step {step_num}: {e}")
                    logger.error(f"Multi-step plan: Step {step_num} failed: {e}")
            else:
                # Direct response for unknown agents
                step_results.append(f"Step {step_num}: handled inline")
        
        combined_result = "\n\n---\n\n".join([
            f"**Step {i+1}:**\n{r}" for i, r in enumerate(step_results)
        ])
        
        return {
            "routing_plan": json.dumps(task_plan),
            "active_agent": "multi_step",
            "agent_result": combined_result,
            "plan_step_results": step_results
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

    async def gmail_node(self, state: NexusState) -> dict:
        plan = json.loads(state["routing_plan"])
        task = plan.get("task", state["user_input"])
        result = await self.gmail_agent.run(task, state["user_context"])
        return {"agent_result": result, "action_taken": True}

    async def direct_response_node(self, state: NexusState) -> dict:
        tz = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        
        filled_prompt = (
            NEXUS_SYSTEM_PROMPT
            .replace("{user_name}", settings.user_name)
            .replace("{user_interests}", settings.user_interests)
            .replace("{user_goals}", settings.user_goals)
            .replace("{user_location}", settings.user_location)
            .replace("{user_context}", state["user_context"])
            .replace("{recalled_context}", state["recalled_memory"])
            .replace("{session_history}", state.get("session_history", ""))
            .replace("{datetime}", now_ist)
        )
        
        response = await self.gemini_client.generate(
            system_prompt=filled_prompt,
            user_message=state["user_input"]
        )
        return {"agent_result": response, "action_taken": False}

    async def multi_step_passthrough_node(self, state: NexusState) -> dict:
        """For multi-step plans, the result is already computed in orchestrate_node."""
        return {"action_taken": True}

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
                text=f"Agent ({agent}) result: {result[:500]}",
                type=f"{agent}_result",
                source="orchestrator",
                importance=0.3
            )
        
        # Check for goal milestones via Groq (fast + free, always-on)
        if len(state['user_input']) > 25:
            try:
                conversation = f"User: {state['user_input']}\nNEXUS: {result[:300]}"
                milestone = await self.goal_tracker.check_for_milestone(conversation)
                if milestone:
                    celebration = milestone.get("celebration_message", "")
                    if celebration and result:
                        return {
                            "agent_result": f"{result}\n\n🎯 {celebration}",
                            "action_taken": state.get("action_taken", False)
                        }
            except Exception as e:
                logger.error(f"Goal tracking in store_memory failed: {e}")
        
        return {"action_taken": state.get("action_taken", False)}

    async def format_response_node(self, state: NexusState) -> dict:
        result = state["agent_result"]
        agent = state["active_agent"]
        
        # For direct responses or short results, skip formatting entirely
        if agent == "direct" or len(result) < 1500:
            return {"final_response": result[:1500]}
        
        # Use Groq for fast formatting of long outputs (free + instant)
        format_prompt = (
            f"Format this as a WhatsApp message for {settings.user_name}. "
            f"Max 1500 chars. Use line breaks. Prioritize important info. "
            f"Natural tone, no corporate-speak."
        )
        
        formatter = self.groq_client if self.groq_client.is_available else self.gemini_client
        
        try:
            formatted = await formatter.generate(
                system_prompt=format_prompt,
                user_message=f"Agent: {agent}\nContent:\n{result}",
                temperature=0.3
            )
            return {"final_response": formatted[:1500]}
        except Exception as e:
            logger.error(f"Response formatting failed: {e}. Using raw text.")
            return {"final_response": result[:1500]}

    async def save_session_node(self, state: NexusState) -> dict:
        """Save user and assistant messages to session after response is formatted."""
        await self.session_manager.add_message(
            state["from_number"], "user", state["user_input"]
        )
        await self.session_manager.add_message(
            state["from_number"], "assistant", state["final_response"][:500]
        )
        # LangGraph requires every node to return at least one state key
        return {"action_taken": state.get("action_taken", False)}

    def _build_graph(self):
        graph = StateGraph(NexusState)

        # Define graph nodes
        graph.add_node("load_session", self.load_session_node)
        graph.add_node("recall_memory", self.recall_memory_node)
        graph.add_node("build_context", self.build_context_node)
        graph.add_node("plan_task", self.plan_task_node)
        graph.add_node("orchestrate", self.orchestrate_node)
        graph.add_node("researcher", self.researcher_node)
        graph.add_node("coder", self.coder_node)
        graph.add_node("scheduler", self.scheduler_node)
        graph.add_node("communicator", self.communicator_node)
        graph.add_node("gmail", self.gmail_node)
        graph.add_node("direct_response", self.direct_response_node)
        graph.add_node("multi_step", self.multi_step_passthrough_node)
        graph.add_node("store_memory", self.store_memory_node)
        graph.add_node("format_response", self.format_response_node)
        graph.add_node("save_session", self.save_session_node)

        # Define edge routing
        graph.set_entry_point("load_session")
        graph.add_edge("load_session", "recall_memory")
        graph.add_edge("recall_memory", "build_context")
        graph.add_edge("build_context", "plan_task")
        graph.add_edge("plan_task", "orchestrate")

        # Routing decision edge
        graph.add_conditional_edges(
            "orchestrate",
            lambda state: state["active_agent"],
            {
                "researcher": "researcher",
                "coder": "coder",
                "scheduler": "scheduler",
                "communicator": "communicator",
                "gmail": "gmail",
                "direct": "direct_response",
                "multi_step": "multi_step"
            }
        )

        # Connect all specialist agents back to memory storage
        for agent_node in ["researcher", "coder", "scheduler", "communicator", "gmail", "direct_response", "multi_step"]:
            graph.add_edge(agent_node, "store_memory")

        graph.add_edge("store_memory", "format_response")
        graph.add_edge("format_response", "save_session")
        graph.add_edge("save_session", END)

        return graph.compile()

    async def process_message(self, user_input: str, from_number: str) -> str:
        if from_number != settings.user_whatsapp_number:
            logger.warning(f"Blocked message from unauthorized number: {from_number}")
            return "Unauthorized"
            
        initial_state: NexusState = {
            "user_input": user_input,
            "from_number": from_number,
            "session_history": "",
            "recalled_memory": "",
            "user_context": "",
            "task_plan": "{}",
            "plan_step_results": [],
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
