"""
Setup configuration for demand-forecast-xai package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="demand-forecast-xai",
    version="0.1.0",
    author="Anderson Lessing",
    description="Demand forecasting with explainable AI (xAI) using M5 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=3.0.0",
        "numpy>=2.0.0",
        "scikit-learn>=1.8.0",
        "lightgbm>=4.6.0",
        "shap>=0.45.0",
        "lime>=0.2.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
)
