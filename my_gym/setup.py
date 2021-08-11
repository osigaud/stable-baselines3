from setuptools import find_packages, setup

setup(
    name="my_gym",
    packages=[package for package in find_packages() if package.startswith("my_gym")],
    version="0.0.1",
    install_requires=["gym"],  # And any other dependencies foo needs
)
