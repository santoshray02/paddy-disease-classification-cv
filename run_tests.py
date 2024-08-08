import pytest
import os
import sys

if __name__ == "__main__":
    # Default arguments
    args = ["tests/"]

    # Check if coverage report is requested
    if "--coverage" in sys.argv:
        args = [
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
        ] + args
        sys.argv.remove("--coverage")

    # Run tests
    pytest.main(args)

    if "--coverage" in args:
        print("\nCoverage report generated in 'coverage_report' directory.")
        print("To view the report, open 'coverage_report/index.html' in a web browser.")
    else:
        print("\nTests completed. Run with --coverage to generate a coverage report.")
