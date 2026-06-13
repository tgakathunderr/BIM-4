from setuptools import setup, find_packages

setup(
    name="bim4",
    version="1.0.0",
    description="A purely biological, CPU-native continuous learning substrate.",
    author="UnikAI Lab",

    packages=["bim4"],
    package_dir={"bim4": "."},
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.57.0",
        "rich>=13.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
