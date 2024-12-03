.PHONY: check-flake8 check-black check-isort check-mypy check-pylint run-black run-isort check format test pack-dataset unpack-dataset
.DEFAULT_GOAL := test

# Checks and Formatting

check-ruff:
	poetry run ruff check

check-mypy:
	poetry run mypy .

check: check-ruff check-mypy

format:
	ruff format

test:
	poetry run pytest

# Run the tools

pack-dataset:
	poetry run python ada_eval/scripts/pack_datasets.py

unpack-dataset:
	poetry run python ada_eval/scripts/unpack_datasets.py
