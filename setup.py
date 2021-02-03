from distutils.core import setup

setup(
    name="parapred-pytorch",
    version="0.1.0",
    author = "Alchemab",
    author_email = "jin@alchemab.com",
    description="PyTorch implementation of Parapred",
    packages=["parapred"],
    package_dir={
        "parapred": "parapred/"
    }
)