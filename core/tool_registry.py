from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    handler: Any  # Agent instance
    triggers: list[str] = field(default_factory=list)
    enabled: bool = True
    version: str = "1.0.0"


class ToolRegistry:
    """
    Plugin-style agent registration system.
    Allows dynamic registration and lookup of agent tools.
    """
    
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        logger.info("ToolRegistry initialized.")
    
    def register(self, tool: Tool):
        """Register a new tool/agent."""
        self._tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name} (v{tool.version}) — {tool.description}")
    
    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Tool unregistered: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        tool = self._tools.get(name)
        if tool and tool.enabled:
            return tool
        return None
    
    def list_tools(self) -> list[dict]:
        """Return list of registered tools with metadata."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "triggers": t.triggers,
                "enabled": t.enabled,
                "version": t.version
            }
            for t in self._tools.values()
        ]
    
    def find_by_trigger(self, text: str) -> Optional[Tool]:
        """Find a tool by matching trigger keywords against text."""
        text_lower = text.lower()
        for tool in self._tools.values():
            if not tool.enabled:
                continue
            for trigger in tool.triggers:
                if trigger.lower() in text_lower:
                    return tool
        return None
    
    def get_tools_description(self) -> str:
        """Generate a description of all tools for prompt injection."""
        lines = []
        for t in self._tools.values():
            if t.enabled:
                triggers_str = ", ".join(t.triggers[:5]) if t.triggers else "N/A"
                lines.append(f"- {t.name}: {t.description} (triggers: {triggers_str})")
        return "\n".join(lines)
