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


setup(name="resnet_base",
      version=get_version(),
      entry_points={"console_scripts": ["resnet_base = resnet_base.__main__:main"]},
      packages=['resnet_base'],
      description="ResNet for TinyImageNet classification",
      install_requires=get_install_requires(),
      author="Timo Denk, Florian Pfisterer, Samed Güner")
