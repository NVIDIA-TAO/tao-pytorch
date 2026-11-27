TAO_CORE ?= ../tao-core
TAO_DATA_SERVICES ?= ../tao-data-services

.PHONY: dev test wheels
dev:
	uv venv --allow-existing .venv
	uv pip install --python .venv -e $(TAO_CORE)/khai -e $(TAO_DATA_SERVICES)/khai_data_services -e '.[dev]'

test:
	.venv/bin/python -m pytest

wheels:
	uv build --wheel --out-dir dist .
	uv build --wheel --out-dir dist $(TAO_CORE)/khai
	uv build --wheel --out-dir dist $(TAO_DATA_SERVICES)/khai_data_services
