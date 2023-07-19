import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="async_agent",
    version="0.0.1",
    author="Alen Rubilar",
    author_email="lclc.alen@gmail.com",
    description="An async runtime for language agents",
    keywords="async agent runtime langchain",
    packages=["async_agent"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
    install_requires=[
        "langchain",
        "pydantic",
        "colorama",
        "aioconsole",
        "python-dotenv",
        "pyee",
        "human-id"
    ]
)
