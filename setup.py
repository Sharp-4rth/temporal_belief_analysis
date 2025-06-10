from setuptools import setup, find_packages

setup(
    name="temporal-belief-analysis",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "pandas>=1.4.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.1.0",
    ]
)