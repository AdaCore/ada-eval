.PHONY: check-flake8 check-black check-isort check-mypy check-pylint run-black run-isort check format test pack-dataset unpack-dataset
.DEFAULT_GOAL := test

# Checks and Formatting

check-ruff:
	uvx ruff check

check-mypy:
	uv run mypy .

check: check-ruff check-mypy

format:
	uvx ruff format

test:
	uv run pytest

# Run the tools

pack-dataset:
	uv run ada_eval pack

unpack-dataset:
	uv run ada_eval unpack
