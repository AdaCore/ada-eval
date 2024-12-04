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
	uv run python ada_eval/scripts/pack_datasets.py

unpack-dataset:
	uv run python ada_eval/scripts/unpack_datasets.py
