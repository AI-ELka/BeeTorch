from setuptools import find_packages, setup

setup(
    name='scitorch',
    packages=find_packages(include=['scitorch']),
    version='0.1.0',
    description='Handle well small models',
    author='AcePeaX',
    install_requires=['pytorch'],
)