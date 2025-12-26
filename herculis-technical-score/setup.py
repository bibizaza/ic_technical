"""Setup file for herculis-technical-score package."""

from setuptools import setup, find_packages

setup(
    name="herculis-technical-score",
    version="1.0.0",
    description="Technical score computation module for Herculis Partners",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
    ],
)
