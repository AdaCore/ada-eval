.PHONY: check-flake8 check-black check-isort check-mypy check-pylint run-black run-isort check format test pack-dataset unpack-dataset
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

# Run the tools

pack-dataset:
	uv run ada-eval pack

unpack-dataset:
	uv run ada-eval unpack

generate-dummy:
	uv run ada-eval generate \
		--tool shell_script \
		--tool-config-file tools/configs/shell_dummy.json

generate-spark-claude:
	uv run ada-eval generate \
		--tool shell_script \
		--tool-config-file tools/configs/claude_code_no_mcp.json

evaluate:
	uv run ada-eval evaluate

evaluate-canonical:
	uv run ada-eval evaluate --canonical
	uv run ada-eval pack

check-datasets:
	uv run ada-eval check-datasets
