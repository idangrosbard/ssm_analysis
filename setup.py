from setuptools import setup, find_packages

setup(
    name="ssm_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=["pyrallis", 'submitit'],
)
