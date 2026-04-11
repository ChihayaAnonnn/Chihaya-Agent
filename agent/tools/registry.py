"""Tool registry for dynamic tool management."""

import logging
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        tool = self._tools.get(name)
        if not tool:
            logger.error("[TOOL_ERROR] tool not found: '%s' (registered: %s)", name, list(self._tools))
            return f"Error: Tool '{name}' not found"
        try:
            errors = tool.validate_params(params)
            if errors:
                logger.warning(
                    "[TOOL_ERROR] invalid params for '%s': %s",
                    name,
                    "; ".join(errors),
                )
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            return await tool.execute(**params)
        except Exception as e:
            logger.error("[TOOL_ERROR] exception in '%s': %s", name, e, exc_info=True)
            return f"Error executing {name}: {str(e)}"

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
