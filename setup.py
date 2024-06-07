from setuptools import setup, find_packages

setup(
    name="pytftk",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "pynvml",
        "colorama",
    ],  # List any dependencies your package needs
    author="Massimiliano Altieri",
    author_email="massimiliano.altieri@uniba.it",
    description="A Python toolkit to develop deep learning models with TensorFlow.",
    url="https://github.com/m-altieri/pytftk",
)
