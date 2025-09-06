"""
Setup script for Cow Lumpy Disease Detection project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cow-lumpy-disease-detection",
    version="1.0.0",
    author="Yash Sakariya",
    author_email="your.email@example.com",
    description="A deep learning project for detecting lumpy skin disease in cattle using CNN",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cow-lumpy-disease-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cow-disease-train=train:main",
            "cow-disease-predict=src.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "*.md"],
    },
    keywords="deep-learning, computer-vision, cnn, cattle, disease-detection, tensorflow, keras",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cow-lumpy-disease-detection/issues",
        "Source": "https://github.com/yourusername/cow-lumpy-disease-detection",
        "Documentation": "https://yourusername.github.io/cow-lumpy-disease-detection/",
    },
)
