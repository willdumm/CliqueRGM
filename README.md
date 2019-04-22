# CliqueRGM

The CliqueRGM Python 3 package implements the tools necessary to build, sample from, and fit an Exponential Random Graph Model (ERGM). This package was written specifically to explore the implications of incorporating maximal cliques as subgraphs of interest in ERGMs for my master's thesis.

## Installing
This package is not on PyPI. To install, clone the repository to your machine (anywhere will do), enter the cliquergm directory (the one which contains this README file), and run

```console
$ ./install.sh
```

This first ensures that the package is uninstalled, and then installs the cliquergm package in editable mode. This way, the package code may be edited in place, with changes taking effect without having to run the install script again.

## Building the Documentation
Documentation is built with Sphinx. To build it, [install Sphinx](http://www.sphinx-doc.org/en/master/usage/installation.html) for Python 3.

In the Terminal, open the ``cliquergm/doc`` folder, and run
```console
$ make html
```
Hopefully everything will work, and you may then open the file ``cliquergm/doc/build/html/index.html`` in a browser to view the html documentation.

You may also run ``make latexpdf`` if you have LaTeX installed, to build a pdf version of the documentation, which you'll then find at ``cliquergm/doc/build/latex/cliquergm.pdf``

## Running Tests
Tests are handled by Pytest. If Pytest is installed, then you may simply run ``pytest`` from within the ``cliquergm`` directory. Tests are located in ``cliquergm/tests``.