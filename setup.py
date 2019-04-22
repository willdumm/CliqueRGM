#!/usr/local/bin/python3
from distutils.core import setup

setup(
    name='cliquergm',
    version='0.1',
    packages=['cliquergm'],
    py_modules=['model', 'graph', 'statistic'],
    package_dir={'': 'lib'},
    install_requires=[],
    author="Will Dumm",
    url='https://github.com/breecummins/WillGraphs'
)
