from distutils.core import setup

setup(
    name="parapred-pytorch",
    version="0.1.0",
    author = "Jinwoo Leem",
    author_email = "jin@alchemab.com",
    packages=["parapred"],
    package_dir={
        "parapred": "parapred/"
    }
)