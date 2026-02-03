.PHONY: install test lint

install:
	poetry install

test:
	poetry run pytest -q

lint:
	poetry run ruff check src tests
