.PHONY: check-flake8 check-black check-isort check-mypy check-pylint run-black run-isort check format test test-all pack-dataset unpack-dataset
.DEFAULT_GOAL := test

# Checks and Formatting

check-ruff:
	uvx ruff check

check-mypy:
	uv run mypy

check: check-ruff check-mypy

format:
	uvx ruff check --select I --fix .
	uvx ruff format

test:
	uv run pytest

# Run all tests, including ones marked as slow
test-all:
	uv run pytest -m ''

# Run the tools

pack-dataset:
	uv run ada-eval pack

unpack-dataset:
	uv run ada-eval unpack

generate-dummy:
	uv run ada-eval generate \
		--tool-config tools/configs/shell_dummy.json -j8

generate-spark-claude-shell:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_code_no_mcp.json -j8

generate-spark-claude:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent.json -j8

generate-spark-claude-haiku-mcp:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent_haiku_mcp.json -j8

generate-spark-claude-sonnet-mcp:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent_sonnet_mcp.json -j8

generate-spark-claude-opus-mcp:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent_opus_mcp.json -j8

generate-spark-claude-haiku:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent_haiku.json -j8

generate-spark-claude-sonnet:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent_sonnet.json -j8

generate-spark-claude-opus:
	uv run ada-eval generate \
		--tool-config tools/configs/claude_agent_opus.json -j8

evaluate:
	uv run ada-eval evaluate
	uv run ada-eval report

evaluate-canonical:
	uv run ada-eval evaluate --canonical
	uv run ada-eval pack

check-datasets:
	uv run ada-eval check-datasets
