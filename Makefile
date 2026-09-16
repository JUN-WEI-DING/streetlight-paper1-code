.PHONY: install demo test
install:
	uv sync --locked --extra pv-benchmark

demo:
	uv run --locked python examples/synthetic/run_demo.py

test:
	uv run --locked --extra pv-benchmark pytest -q
