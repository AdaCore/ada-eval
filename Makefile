.PHONY: check-flake8 check-black check-isort check-mypy check-pylint run-black run-isort check format test pack-dataset unpack-dataset
.DEFAULT_GOAL := test

# Checks and Formatting

check-flake8:
	poetry run flake8

check-black:
	poetry run black --check .

check-isort:
	poetry run isort --check --diff .

check-mypy:
	poetry run mypy .

check-pylint:
	poetry run pylint .

run-black:
	poetry run black .

run-isort:
	poetry run isort .

check: check-flake8 check-isort check-black check-mypy check-pylint

format: run-black run-isort

test:
	poetry run pytest

# Run the tools

pack-dataset:
	poetry run python ada_eval/scripts/pack_datasets.py

unpack-dataset:
	poetry run python ada_eval/scripts/unpack_datasets.py
