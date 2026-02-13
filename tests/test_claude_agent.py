"""Tests for the ClaudeAgent tool."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self
from unittest.mock import MagicMock, patch

import pytest

from ada_eval.datasets.types import ExitStatus
from ada_eval.tools import ClaudeAgent
from ada_eval.tools.claude_agent import ClaudeAgentConfig


@pytest.fixture
def claude_agent_config() -> ClaudeAgentConfig:
    """Create a basic ClaudeAgentConfig for testing."""
    return ClaudeAgentConfig(timeout_s=60, max_budget_usd=1.0)


@pytest.fixture
def mock_spark_sample(tmp_path: Path) -> MagicMock:
    """Create a mock SparkSample for testing."""
    # Create a minimal directory structure
    sample_dir = tmp_path / "test_sample"
    sample_dir.mkdir()
    (sample_dir / "src").mkdir()
    (sample_dir / "src" / "test.adb").write_text("-- Ada code")

    sample = MagicMock()
    sample.name = "test_sample"
    sample.prompt = "Add a postcondition to ensure the result is positive"

    # Use a context manager that returns the sample_dir
    @contextmanager
    def mock_unpacked():
        yield sample_dir

    sample.sources.unpacked = mock_unpacked
    sample.model_dump.return_value = {
        "name": "test_sample",
        "prompt": "Add a postcondition to ensure the result is positive",
        "location": {"path": "src/test.adb", "subprogram_name": "Test"},
        "comments": "",
        "sources": {},
        "canonical_solution": {},
        "canonical_evaluation_results": [],
        "required_checks": [],
        "unit_tests": {},
    }
    return sample


@dataclass
class MockResultMessage:
    """Mock ResultMessage from the SDK."""

    is_error: bool = False
    total_cost_usd: float | None = 0.05
    usage: dict[str, Any] | None = None
    num_turns: int = 3
    session_id: str = "test_session"

    def __post_init__(self):
        if self.usage is None:
            object.__setattr__(
                self, "usage", {"input_tokens": 100, "output_tokens": 50}
            )


@dataclass
class MockTextBlock:
    """Mock TextBlock from the SDK."""

    text: str
    type: str = "text"


@dataclass
class MockToolUseBlock:
    """Mock ToolUseBlock from the SDK."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)
    type: str = "tool_use"


@dataclass
class MockAssistantMessage:
    """Mock AssistantMessage from the SDK."""

    content: list[Any]


class MockClaudeSDKClient:
    """Mock ClaudeSDKClient that's a proper async context manager."""

    def __init__(
        self, response_messages: list[Any], should_raise: Exception | None = None
    ):
        self.response_messages = response_messages
        self.should_raise = should_raise
        self.query_called = False
        self.prompt_received: str | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def query(self, prompt: str) -> None:
        self.query_called = True
        self.prompt_received = prompt

    async def interrupt(self) -> None:
        pass

    async def receive_response(self) -> AsyncIterator[Any]:
        if self.should_raise:
            raise self.should_raise
        for msg in self.response_messages:
            yield msg


class TestClaudeAgentConfig:
    """Tests for ClaudeAgentConfig."""

    def test_from_file(self, tmp_path: Path):
        """Test loading config from file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"tool": "claude_agent", "timeout_s": 120, "max_budget_usd": 2.0}'
        )

        config = ClaudeAgentConfig.from_file(config_file)

        assert config.timeout_s == 120
        assert config.max_budget_usd == 2.0
        assert config.model is None

    def test_from_file_with_model(self, tmp_path: Path):
        """Test loading config with model override."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"tool": "claude_agent", "timeout_s": 60, '
            '"model": "claude-sonnet-4-20250514"}'
        )

        config = ClaudeAgentConfig.from_file(config_file)

        assert config.model == "claude-sonnet-4-20250514"

    def test_from_file_with_mcp_config(self, tmp_path: Path):
        """Test loading config with mcp_config path resolution."""
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text('{"mcpServers": {}}')

        config_file = tmp_path / "configs" / "config.json"
        config_file.parent.mkdir()
        config_file.write_text(
            '{"tool": "claude_agent", "timeout_s": 60, "mcp_config": "../.mcp.json"}'
        )

        config = ClaudeAgentConfig.from_file(config_file)

        assert config.mcp_config is not None
        assert config.mcp_config == mcp_file.resolve()
        assert config.mcp_config.exists()

    def test_from_file_without_mcp_config(self, tmp_path: Path):
        """Test loading config without mcp_config leaves it as None."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"tool": "claude_agent", "timeout_s": 60}')

        config = ClaudeAgentConfig.from_file(config_file)

        assert config.mcp_config is None

    def test_invalid_tool_type(self, tmp_path: Path):
        """Test that wrong tool type raises error."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"tool": "shell_script", "timeout_s": 60}')

        with pytest.raises(ValueError, match="Expected tool type 'claude_agent'"):
            ClaudeAgentConfig.from_file(config_file)


class TestClaudeAgent:
    """Tests for ClaudeAgent."""

    def test_create_from_config_file(self, tmp_path: Path):
        """Test creating agent from config file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            '{"tool": "claude_agent", "timeout_s": 120, "max_budget_usd": 1.5}'
        )

        agent = ClaudeAgent.from_config_file(config_file)

        assert agent.config.timeout_s == 120
        assert agent.config.max_budget_usd == 1.5

    @contextmanager
    def _patch_sdk(self, mock_client: MockClaudeSDKClient):
        """Patch SDK names in the claude_agent module so isinstance checks work."""
        mod = "ada_eval.tools.claude_agent"
        with (
            patch(f"{mod}.ClaudeSDKClient", MagicMock(return_value=mock_client)),
            patch(f"{mod}.AssistantMessage", MockAssistantMessage),
            patch(f"{mod}.ResultMessage", MockResultMessage),
            patch(f"{mod}.TextBlock", MockTextBlock),
        ):
            yield

    def test_apply_success(
        self,
        claude_agent_config: ClaudeAgentConfig,
        mock_spark_sample: MagicMock,
    ):
        """Test successful agent run."""
        assistant_msg = MockAssistantMessage(
            content=[MockTextBlock(text="I'll analyze the code...")]
        )
        result_msg = MockResultMessage()

        mock_client = MockClaudeSDKClient([assistant_msg, result_msg])

        with self._patch_sdk(mock_client):
            agent = ClaudeAgent(claude_agent_config)
            result = agent.apply(mock_spark_sample)

            assert result.generation_stats.exit_status == ExitStatus.SUCCESS
            assert result.generation_stats.total_cost_usd == 0.05
            assert result.generation_stats.num_turns == 3

    def test_apply_timeout(
        self,
        mock_spark_sample: MagicMock,
    ):
        """Test timeout handling."""

        class SlowMockClient(MockClaudeSDKClient):
            """Mock client that sleeps during receive_response."""

            async def receive_response(self) -> AsyncIterator[Any]:
                await asyncio.sleep(5)
                yield MagicMock()

        mock_client = SlowMockClient([])

        with self._patch_sdk(mock_client):
            config = ClaudeAgentConfig(timeout_s=1)
            agent = ClaudeAgent(config)
            result = agent.apply(mock_spark_sample)

            assert result.generation_stats.exit_status == ExitStatus.TIMEOUT

    def test_apply_sdk_error(
        self,
        claude_agent_config: ClaudeAgentConfig,
        mock_spark_sample: MagicMock,
    ):
        """Test SDK error handling."""
        mock_client = MockClaudeSDKClient([], should_raise=RuntimeError("SDK error"))

        with self._patch_sdk(mock_client):
            agent = ClaudeAgent(claude_agent_config)
            result = agent.apply(mock_spark_sample)

            assert result.generation_stats.exit_status == ExitStatus.SDK_ERROR

    def test_apply_budget_exceeded(
        self,
        mock_spark_sample: MagicMock,
    ):
        """Test budget exceeded detection."""
        result_msg = MockResultMessage(total_cost_usd=1.5)
        mock_client = MockClaudeSDKClient([result_msg])

        with self._patch_sdk(mock_client):
            config = ClaudeAgentConfig(timeout_s=60, max_budget_usd=1.0)
            agent = ClaudeAgent(config)
            result = agent.apply(mock_spark_sample)

            assert result.generation_stats.exit_status == ExitStatus.BUDGET_EXCEEDED
