import json
from pathlib import Path

from .claude_agent import ClaudeAgent
from .shell_script import ShellScript
from .tool_type import ToolType


class UnsupportedToolError(Exception):
    def __init__(self, tool: str):
        super().__init__(f"Unsupported tool: {tool}")


def create_tool_from_config(config_file: Path) -> ShellScript | ClaudeAgent:
    """
    Create a tool instance from a config file.

    The tool type is inferred from the 'tool' field in the config.

    Args:
        config_file (Path): path to config file for the tool. should be a JSON
            file containing at least a 'tool' field indicating the type of tool.

    Raises:
        UnsupportedToolError: if the tool type specified in the config is not supported.

    Returns:
        ShellScript | ClaudeAgent: the instantiated tool object

    """
    # Peek at the config to determine tool type
    config_data = json.loads(config_file.read_text(encoding="utf-8"))
    tool_type = ToolType(config_data.get("tool", "shell_script"))

    match tool_type:
        case ToolType.SHELL_SCRIPT:
            return ShellScript.from_config_file(config_file)
        case ToolType.CLAUDE_AGENT:
            return ClaudeAgent.from_config_file(config_file)
        case _:
            raise UnsupportedToolError(tool_type)
