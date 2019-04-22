*******
Fitting
*******
.. default-role:: code

The :meth:`~cliquergm.model.fit()` function can be used to find parameters
which approximate a desired subgraph count distribution. For example,

>>> ips = {'Cliques': [0]*8, 'Edge': 0}  # Initial Parameters
>>> target_counts = {'Cliques': [0, 1, 3, 5, 6, 5, 2, 0]}
>>> bounds = {'Cliques': [(0,0), None, (-1, 2), (-1, 2), (-1, 2), (-1, 2)]}
>>> params = fit(target_counts, 20, .3, .05, ips=ips, bounds=bounds)

Choosing Fitting Parameters
===========================

The fit function must be supplied with either initial parameters (`ips`) or
bounds, although both are allowed.

First, determine target subgraph counts:

>>> target_counts = {'Cliques': [0,1,3,5,2,0,0,0], 'Edge': 10}

For statistics which require a vector of counts, like cliques, the vector must
be exactly the right length. That is, when sampling graphs with 15 nodes, the
target clique count vector must have length 15.

If initial parameters are not supplied, then the best initial parameter set for
the fitting algorithm will be chosen from a randomly selected list of random
parameter sets within the given bounds. Bounds not supplied will be assumed
fixed at a value of `0`. Bounds supplied may be a tuple range or `None`. The
number of random initial parameters checked can be controlled by the keyword
argument `random_samples`.

>>> bounds = {'Cliques': [(-.5, .5)]*6}
>>> model.fit(target_counts, 8, 0.1, 0.05, bounds=bounds, random_samples=50)

In this example, the inferred bounds will be `{'Cliques': [(-.5, .5), (-.5, .5),
(-.5, .5), (-.5, .5), (-.5, .5), (-.5, .5), (0, 0), (0, 0)], 'Edge': (0, 0)}`.

If initial parameters are supplied, then the bounds search step will be skipped,
and the fitting algorithm will initiate with supplied parameters. In this case,
bounds are not required, and will be inferred from the supplied initial
parameters. Any subgraph parameter supplied which does not have an associated
bound will be assigned an unrestricted bound of `None`. Bounds for parameters
not supplied will be fixed at `(0,0)`. Any supplied bounds will be respected.

>>> bounds = {'Cliques': [(-0.5, 0.5)]}
>>> initial_params = {'Cliques':[1, -.2, 0]}
>>> model.fit(target_counts, 8, 0.1, 0.5, ips=initial_params, bounds=bounds)

In this example, the inferred bounds will be `{'Cliques': [(-0.5, 0.5), None,
None, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], 'Edge': (0, 0)}`

Avoiding degenerate regions in parameter space
==============================================
Sometimes the fitting algorithm can get stuck in degenerate regions of
parameter space. For example, suppose we attempt to fit a target count
distribution on an 10-node graph using initial parameters `{'Cliques': [0]*10}`.

If the target count distribution specifies a large number of 6-cliques, the
fitting algorithm may move to a parameter set such as
`{'Cliques': [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]}`. Since convergence of sampling
is measured by initializing chains from both an empty and a complete graph,
a parameter set like this one is likely to never converge; the chain starting
from the empty graph will never develop a 6-clique, and will act exactly as if
all parameter were zero, while the chain initialized with the complete graph
will decay as if its parameters were all zero until 6-cliques develop.
Because of the large parameter for 6-cliques, this second chain will preserve
6-cliques and the two chains will never look alike.

One method of avoiding this behavior is to run the fitting algorithm many times,
constraining all but a few parameters each time. When fitting graphs based
on a clique count distribution, this could mean restricting all but 1- and
2-clique parameters the first time the fitting algorithm is run, then
iteratively applying the resulting parameters as initial parameters for
subsequent runs, increasing the number of free clique parameters each time.