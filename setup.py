from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"

setup(
    name="gefs-temperature-bias-correction",
    version="0.1.0",
    description="GEFS 2-m temperature mean-bias correction and verification workflow",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Youkyoung Jang",
    packages=find_packages(include=["gefs_bias_correction", "gefs_bias_correction.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0,<3.0",
        "xarray>=2024.1.0",
        "matplotlib>=3.7",
        "requests>=2.31",
        "herbie-data>=2025.2.0",
        "cfgrib>=0.9.10",
        "eccodes>=2.37.0",
        "s3fs>=2024.1.0",
        "fsspec>=2024.1.0",
        "scikit-learn>=1.3",
        "netCDF>=1.6",
    ],
    extras_require={
        "plots": ["cartopy>=0.23"],
        "dev": ["jupyter", "ipykernel", "pytest"],
    },
)
