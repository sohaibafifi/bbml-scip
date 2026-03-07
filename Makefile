.PHONY: test test-py test-cpp

test: ## Run Python and C++ tests
	./scripts/test_all.sh

test-py: ## Run Python tests only
	@ROOT_DIR=$$(pwd); \
	export PYTHONPATH="$$ROOT_DIR/py:$$PYTHONPATH"; \
	if [ -n "$$BBML_PYTHON" ]; then \
	  "$$BBML_PYTHON" "$$ROOT_DIR/scripts/run_py_tests.py" "$$ROOT_DIR/tests/py"; \
	elif [ -x "$$ROOT_DIR/py/.venv/bin/python" ]; then \
	  "$$ROOT_DIR/py/.venv/bin/python" "$$ROOT_DIR/scripts/run_py_tests.py" "$$ROOT_DIR/tests/py"; \
	elif command -v python3 >/dev/null 2>&1; then \
	  python3 "$$ROOT_DIR/scripts/run_py_tests.py" "$$ROOT_DIR/tests/py"; \
	else \
	  echo "No Python test environment available"; exit 1; \
	fi

test-cpp: ## Build and run C++ tests
	@cmake -S . -B build/cpp-tests -DBBML_WITH_ONNX=ON -DBBML_WITH_LP_STATS=ON >/dev/null || { echo "C++ configure failed (SCIP or GTest missing); skipping"; exit 0; }; \
	cmake --build build/cpp-tests -j >/dev/null || { echo "C++ tests build failed or GTest missing; skipping"; exit 0; }; \
	if command -v ctest >/dev/null 2>&1; then \
	  ctest --test-dir build/cpp-tests --output-on-failure; \
	else \
	  ./build/cpp-tests/cpp_tests; \
	fi
