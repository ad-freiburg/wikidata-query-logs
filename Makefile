.PHONY: format check lint type-check mypy-check pyright-check test all clean help

# Default target
all: format check

help:
	@echo "Available targets:"
	@echo "  format         - Format code with ruff"
	@echo "  check          - Check code style with ruff"
	@echo "  check-all      - Check with all ruff rules"
	@echo "  lint           - Alias for check"
	@echo "  type-check     - Run both mypy and pyright type checking"
	@echo "  mypy-check     - Run mypy type checking only"
	@echo "  pyright-check  - Run pyright type checking only"
	@echo "  test           - Run pytest tests"
	@echo "  all            - Run format and check (default)"
	@echo "  clean          - Remove cache files"

# Format code with ruff
format:
	@echo "Formatting Python files with ruff..."
	ruff format *.py

# Check code style with ruff (standard rules)
check:
	@echo "Checking Python files with ruff..."
	ruff check *.py

# Check with all ruff rules (comprehensive)
check-all:
	@echo "Checking Python files with all ruff rules..."
	ruff check --select ALL *.py

# Alias for check
lint: check

# Run mypy type checking only
mypy-check:
	@echo "Running mypy type checking..."
	@mypy *.py --ignore-missing-imports --check-untyped-defs

# Run pyright type checking only
pyright-check:
	@echo "Running pyright type checking..."
	@pyright *.py

# Run both mypy and pyright type checking
type-check: mypy-check pyright-check
	@echo ""
	@echo "✓ Type checking complete (mypy + pyright)"

# Run pytest tests
test:
	@echo "Running pytest..."
	pytest test_*.py -v

# Clean cache files
clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pyright" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cache cleaned."
