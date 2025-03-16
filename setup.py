from setuptools import setup, find_packages

setup(
    name="queensgame",
    version="0.1",
    packages=find_packages(exclude=("test", "docs", "examples")),
)
