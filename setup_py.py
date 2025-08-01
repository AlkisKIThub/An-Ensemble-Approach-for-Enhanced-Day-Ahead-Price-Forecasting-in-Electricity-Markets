"""
Setup script for Day-Ahead Electricity Price Forecasting System
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="day-ahead-epf",
    version="1.0.0",
    author="Alkis Kitsatoglou",
    author_email="",
    description="An Ensemble Approach for Enhanced Day Ahead Price Forecasting in Electricity Markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/day-ahead-epf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "viz": [
            "pandasgui>=0.2.0",
            "pygwalker>=0.1.0",
            "plotly>=5.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "day-ahead-epf=day_ahead_EPF:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords="electricity price forecasting machine learning time series ensemble",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/day-ahead-epf/issues",
        "Source": "https://github.com/yourusername/day-ahead-epf",
        "Documentation": "https://github.com/yourusername/day-ahead-epf/blob/main/README.md",
    },
)