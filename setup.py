from setuptools import setup, find_packages

setup(
    name="SpectralConvxD",
    version="0.1.0",
    author="jarod ketcha",
    author_email="jarod.ketchakouakep@unamur.be",
    description="Convolution Neural Networks in the spectral domain",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DALPHA-sys-24/SpectralConvxD.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "requests>=2.25.0",
        # autres dépendances
    ],
)