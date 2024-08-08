import pytest
import os

if __name__ == "__main__":
    # Run tests and generate coverage report
    pytest.main([
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:coverage_report",
        "tests/"
    ])

    print("\nCoverage report generated in 'coverage_report' directory.")
    print("To view the report, open 'coverage_report/index.html' in a web browser.")
