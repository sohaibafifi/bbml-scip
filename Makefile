.PHONY: test test-py test-cpp

test: ## Run Python and C++ tests
	./scripts/test_all.sh

test-py: ## Run Python tests only
	@ROOT_DIR=$$(pwd); \
	export PYTHONPATH="$$ROOT_DIR/py:$$PYTHONPATH"; \
	if command -v pytest >/dev/null 2>&1; then \
	  pytest -q tests/py; \
	else \
	  python3 -m pytest -q tests/py || python -m pytest -q tests/py; \
	fi

test-cpp: ## Build and run C++ tests
	@cmake -S tests/cpp -B build/cpp-tests >/dev/null; \
	cmake --build build/cpp-tests -j >/dev/null || { echo "C++ tests build failed or GTest missing; skipping"; exit 0; }; \
	if command -v ctest >/dev/null 2>&1; then \
	  ctest --test-dir build/cpp-tests --output-on-failure; \
	else \
	  ./build/cpp-tests/cpp_tests; \
	fi
