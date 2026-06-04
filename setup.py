from setuptools import setup, find_packages

setup(
    name="agent-sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "astropy>=5.0",
        "matplotlib>=3.5",
        "pyyaml>=6.0",
    ],
    extras_require={
        "pylenslib": ["pylenslib>=0.1"],
        "imsim": ["imsim>=0.1"],
        "slsim": ["slsim>=0.1"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "sphinx>=5.0",
        ],
    },
    python_requires=">=3.9",
    author="AGENT Team",
    description="AGENT Simulation: Strong Gravitational Lensing simulations for dark matter studies",
    long_description=open("README.md").read() if __file__ else "",
    long_description_content_type="text/markdown",
)
