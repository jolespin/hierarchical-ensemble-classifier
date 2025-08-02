"""
Setup script for hierarchical_ensemble_classifier package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hierarchical-ensemble-classifier",
    version="0.1.0",
    author="Josh L. Espinoza",
    author_email="jespinoz@jcvi.org",
    description="A scikit-learn compatible hierarchical ensemble classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jolespin/hierarchical-ensemble-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "networkx>=2.5",
        "scipy>=1.6.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
        "skclust": [
            "skclust>=2025.7.26",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=3.5",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
        "all": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "skclust>=2025.7.26",
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
            "sphinx>=3.5",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
    },
)
