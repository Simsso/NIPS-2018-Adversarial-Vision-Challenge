"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from codecs import open
from setuptools import setup


def get_version():
    with open('version.txt') as ver_file:
        version_str = ver_file.readline().rstrip()
    return version_str


def get_install_requires():
    with open('requirements.txt') as reqs_file:
        reqs = [line.rstrip() for line in reqs_file.readlines()]
    return reqs


setup(name="nips_defense",
      version=get_version(),
      entry_points={"console_scripts": ["nips_defense = nips_defense.__main__:main"]},
      packages=['nips_defense'],
      description="ResNet for TinyImageNet classification",
      install_requires=get_install_requires(),
      author="Timo Denk, Florian Pfisterer, Samed Guener")
