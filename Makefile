# Makefile for checking Python project health

PYTHON_FILES=.

.PHONY: check format lint deadcode analyze

check: format lint deadcode analyze
	@echo "✅ All checks complete. You're good to go!"

format:
	@echo "🧼 Formatting with black, isort, and autoflake..."
	black $(PYTHON_FILES)
	isort $(PYTHON_FILES)
	autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive $(PYTHON_FILES)

lint:
	@echo "🔍 Running Ruff for linting..."
	ruff check $(PYTHON_FILES)

deadcode:
	@echo "☠️ Scanning for unused code with Vulture..."
	vulture $(PYTHON_FILES)

analyze:
	@echo "📊 Analyzing complexity with Radon..."
	radon cc $(PYTHON_FILES) -a
	radon mi $(PYTHON_FILES)
