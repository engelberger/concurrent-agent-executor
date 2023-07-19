import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="concurrent_agent_executor",
    version="0.0.1",
    author="Alen Rubilar",
    author_email="lclc.alen@gmail.com",
    description="An concurrent runtime for tool-enhanced language agents",
    keywords="concurrency agent executor runtime langchain",
    packages=find_packages(),
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    install_requires=[
        "openai",
        "langchain",
        "pydantic",
        "colorama",
        "aioconsole",
        "python-dotenv",
        "pyee",
        "human-id",
    ],
)
