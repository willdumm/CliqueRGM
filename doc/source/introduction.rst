************
Introduction
************

The Model class provides functions to set up and sample from an ERGM. A
:class:`~cliquergm.model.Model` object is initialized with a list of initial
graph configurations, for each of which a :class:`~cliquergm.chain.Chain`
object is created and sampled.

Randomness in CliqueRGM is provided by ``numpy.random``. For reproducibility
of results, simply set ``numpy.random.seed()`` before using CliqueRGM functions
and methods.

Model Initialization
--------------------

Initial graph configurations may be generated using the graph generator
functions in the graph module, for example
:meth:`~cliquergm.graph.erdos_renyi_graph()`.

>>> import numpy as np
>>> np.random.seed(88)
>>> from cliquergm.model import Model
>>> import cliquergm.graph as gr
>>> initial_graphs = [gr.erdos_renyi_graph(15, .3 * p) for p in range(4)]

A model must also be initialized with a dictionary of parameters keyed by
statistic names. These are automatically converted to Statistic objects
during Model object initialization. See :doc:`Statistic </reference/statistic>`
subclass names in the statistic module for available keys and parameter formats.


>>> stats = {'Cliques': [.1, .01, -.05, .1, -.1, .1, -.1, -.05], 'Edge': 0 }
>>> model = Model(initial_graphs, stats)


Sampling
--------

A model object may be sampled a fixed number of times using the sample method

>>> model.sample(1000)

or sampled to convergence:

>>> model.sample_converged(1000)

Sampled graphs are stored in the ``sample_list`` attribute of each of the
model's :class:`~cliquergm.chain.Chain` objects. They may be retrieved
directly:

>>> samples = []
>>> for chain in model.chains:
        samples.append(chain.sample_list)


Model Preferences
-----------------
Sampling preferences such as burnin and sample interval may be modified using 
the :meth:`~cliquergm.model.Model.update_prefs()` method:

>>> model.update_prefs(burnin=500, sample_interval=300)


Model Fitting
-------------
See :doc:`Fitting <fitting>`.
