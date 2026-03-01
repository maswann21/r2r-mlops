from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="r2r-coating-mlops",
    version="0.1.0",
    author="R2R Coating MLOps Team",
    description="R2R Coating Defect Detection and Auto-Optimization MLOps System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "mlflow>=2.7.0",
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "psycopg2-binary>=2.9.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "r2r-api=api.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
