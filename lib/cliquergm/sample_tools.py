# Copyright(C) 2019 Will Dumm

# This file is part of the CliqueRGM Python package.
# CliqueRGM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY
# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see < https: // www.gnu.org/licenses/>.

import numpy as np
from scipy.stats import ks_2samp
from random import sample, choice
from cliquergm.statistic import names
import copy


def outersquare(v):
    return(np.outer(v, v))


def dist(entryset, size):
    ''' Takes as argument a set of frozen sets, 2d list, or similar
    generator or iterator object. Returns distribution list of length size
    with [i-1]st entry the number of sublists of length i in entryset.
    '''
    distribution = [0] * size
    for clique in entryset:
        distribution[len(clique) - 1] += 1
    return(distribution)


def issubset_of_element(el, set2d):
    '''
    Returns True if el is a subset of some element of set2d.
    Otherwise returns False.
    '''
    for M in set2d:
        if el.issubset(M):
            return(True)
    return(False)


def ispropersubset_of_element(el, set2d):
    '''
    Returns True if el is a proper subset of some element of set2d.
    Otherwise returns False.
    '''
    for M in set2d:
        if el < M:
            return(True)
    return(False)


def trim_zeros(a):
    '''
    replaces leading zeros with spaces and trims trailing zeros.
    '''
    trimmed = []
    for i in range(len(a) - 1, -1, -1):
        if a[i] != 0:
            break
    for j in range(i+1):
        if a[j] == 0:
            trimmed.append(' ')
        else:
            trimmed.append(a[j])
    return(trimmed)


def pad_zeros(a):
    '''
    replaces all zeros with spaces, return array of strings
    '''
    padded = []
    for i in range(len(a)):
        if a[i] == 0:
            padded.append('0')
        else:
            padded.append(str(a[i]))
    return(padded)


def preap_zeros(a):
    """ Add a leading and trailing zero """
    padded = [0]
    padded.extend([i for i in a])
    padded.append(0)
    return(padded)


def preap_extrap(a):
    """ Extrapolate a list item on either end of list """
    padded = [a[0] - (a[1] - a[0])]
    padded.extend([i for i in a])
    padded.append(a[-1] + (a[-1] - a[-2]))
    return(padded)


def dict_len(d):
    """ Count the length of values in a parameter dictionary """
    c = 0
    for k in d:
        if isinstance(d[k], (list, np.ndarray)):
            c += len(d[k])
        else:
            c += 1
    return(c)


def inbounds(val, bound):
    """ Return if val is in (closed) interval represented by tuple bound,
    always return True if bound is None. """
    if bound is None:
        return(True)
    if val >= bound[0] and val <= bound[1]:
        return(True)
    else:
        return(False)


def setinbounds(pars, bounds):
    """ Return if parameter set is in bounds represented by bounds dictionary"""
    for i in range(pars._inside_len()):
        if not inbounds(pars[i], bounds[i]):
            return(False)
    return(True)


def count_distance(reference, observation, mask=None, failval=np.inf):
    """ Sum for each subgraph the distance between reference and observed
    counts. If mask dictionary is supplied, keys with value False will be
    ignored. If either reference or observation is False, returns np.inf.
    Reference and observation order matters, since function iterates through
    keys in the reference dictionary."""
    if not reference or not observation:
        return(failval)
    c = 0
    # Must index by keys in reference, sometimes we don't care about some
    # counted keys (those for which parameters are being fit, think Edges)
    if mask is None:
        mask = {k: True for k in reference}
    for k in reference:
        if mask[k]:
            if '__len__' in dir(reference[k]):
                for i in range(len(reference[k])):
                    c += abs(reference[k][i] - observation[k][i])
            else:
                c += abs(reference[k] - observation[k])
    return(round(c, 5))


def rand_bound(bound):
    """ Returns a random number (uniform distribution) within tuple bound
    given. If bound is None, returns a value from the standard normal
    distribution about the origin."""
    if bound is None:
        return(np.random.randn())
    else:
        return(np.random.rand() * (bound[1]-bound[0]) + bound[0])


def rand_ips(bounds):
    """ Returns a random parameter set in the bounds given. For a bound of
    None on a parameter, its random value will be chosen from a standard normal
    distribution about 0."""
    ips = bounds.deepcopy()
    for i in range(ips._inside_len()):
        ips[i] = rand_bound(bounds[i])
    ips.makeParamDict()
    return(ips)


def count_dict(sample_list, split=False):
        """Summarize statistic counts of sampled graphs in sample_list in a
        dictionary.

        Returns
        -------
        count_dict : dictionary {str: numpy array}
            Parameter counts are keyed by statistic name, with arrays
            of counts as values.
        """
        d = {key: [] for key in sample_list[0].count_dict(split=split)}
        for g in sample_list:
            dg = g.count_dict(split=split)
            for key in d:
                d[key].append(dg[key])
        for key in d:
            d[key] = np.array(d[key])
        return(ParamDict(d))


def avg_count_dict(sample_list):
    """Returns False if sample_list is empty"""
    if len(sample_list) == 0:
        return(False)
    d = count_dict(sample_list)
    out = {}
    for k in d:
        out[k] = np.mean(d[k], axis=0)
    return(ParamDict(out))


def avg_density(sample_list):
    densities = []
    for g in sample_list:
        densities.append(g.density())
    return(np.mean(densities))


def paired_chains_test(sample1, sample2, p=0.05, return_p=False):
    """A method to assess convergence by comparing subgraph counts in samples.

    Parameters
    ----------

    sample1, sample2 : list
        Lists containing graphs sampled from the same parameter set, but using
        different initial conditions.

    p : float
        (Default ``0.05``)
        The p-value used to assess whether pairs of within-sample and
        between-sample samples of pairwise distances are significantly
        different. Since convergence is assumed only if the Kolmogorov-Smirnov
        test is inconclusive when applied to all three pairs of distance
        samples, a smaller p-value makes the convergence test less strict. That
        is, if ``p = 0``, then this function will always return ``False``,
        indicating convergence.

    return_p : bool
        (Default ``False``)
        Whether the function should return the results from each application
        of the Kolmogorov-Smirnov test.

    Returns
    -------

    not_converged : bool
        ``True`` if sample1 and sample2 seem significantly different, based on
        their subgraph counts. If all three KS tests are inconclusive, returns
        ``False``, indicating that convergence may have occurred.

    results : numpy array
        An array of arrays, each containing the result of the
        ``scipy.stats.ks_2samp()`` function applied to one pair of samples
        of pairwise subgraph count vector distances. Returned only if
        ``return_p`` is ``True``.
    """
    counts1 = [g.count_dict() for g in sample1]
    counts2 = [g.count_dict() for g in sample2]
    # print(avg_count_dict(sample1))
    # print(avg_count_dict(sample2))
    dist1 = [count_distance(*sample(counts1, 2)) for _ in range(1000)]
    dist2 = [count_distance(*sample(counts2, 2)) for _ in range(1000)]
    dist3 = [count_distance(choice(counts1), choice(counts2))
             for _ in range(1000)]
    combos = [[dist1, dist2], [dist2, dist3], [dist1, dist3]]
    results = np.array([ks_2samp(*pair) for pair in combos])
    # print(results)
    if return_p:
        return(results)
    return(min(results[:, 1]) < p)  # That is, at least one test is conclusive


def assert_len(l, n, element):
    """Extends list to length n by appending element. Or, trim list to length n
    """
    while len(l) < n:
        l.append(element)
    while len(l) > n:
        l.pop()


class StatDict(dict):
    """For containing dictionaries which have keys in statistic.names. When
    values will be parameters or counts, use ParamDict subclass"""

    def deepcopy(self):
        return(copy.deepcopy(self))

    def keytuple(self):
        """ Returns tuple of keys in d ordered according to names OrderedDict. """
        lt = []
        for k in names:
            if k in self:
                lt.append(k)
        return(tuple(lt))

    def _mapinside(self, dofunc, reflexive=False):
        """ A function which iterates through keys in the order of
        statistic.names, and if necessary through values which are lists,
        and runs dofunc(val, key, index) for each.

        If reflexive is True, then instead of returning a list of output from
        dofunc, output of dofunc is assigned to each value in self."""
        out = []
        for key in names:
            if key in self:
                if isinstance(self[key], (list, np.ndarray)):
                    for i in range(len(self[key])):
                        if reflexive:
                            self[key][i] = dofunc(self[key][i], key, i)
                        else:
                            out.append(dofunc(self[key][i], key, i))
                else:
                    if reflexive:
                        self[key] = dofunc(self[key], key, None)
                    else:
                        out.append(dofunc(self[key], key, None))
        if not reflexive:
            return(out)

    def makeParamDict(self):
        """ Converts self to an instance of the subclass ParamDict """
        self.__class__ = ParamDict

    def nametuple(self):
        def dofunc(val, key, index):
            if index is None:
                return(key)
            else:
                return("{}-{}".format(index + 1, key))
        return(tuple(self._mapinside(dofunc)))

    def valuetuple(self):
        def dofunc(val, key, index):
            return(val)
        return(tuple(self._mapinside(dofunc)))

    def _dict_ind(self, n, val=None):
        c = 0
        for k in names:
            if k in self:
                if isinstance(self[k], (list, np.ndarray)):
                    for i in range(len(self[k])):
                        if c == n:
                            if val is None:
                                return(self[k][i])
                            else:
                                self[k][i] = val
                                return
                        c += 1
                else:
                    if c == n:
                        if val is None:
                            return(self[k])
                        else:
                            self[k] = val
                            return
                    c += 1
        raise(IndexError)

    def _inside_len(self):
        def dofunc(val, key, index):
            return(1)
        return(sum(self._mapinside(dofunc)))

    def __getitem__(self, index):
        if isinstance(index, int):
            return(self._dict_ind(index))
        else:
            return(super().__getitem__(index))

    def __setitem__(self, index, val):
        if isinstance(index, int):
            self._dict_ind(index, val=val)
        else:
            super().__setitem__(index, val)


class ParamDict(StatDict):
    """For containing dictionaries of subgraph parameters or counts, where keys
    are strings naming statistic subclasses, and values are either lists of
    parameters or a single scalar parameter."""

    def valuetuple(self):
        out = super().valuetuple()
        return(tuple([round(item, 5) for item in out]))

    def __eq__(self, other):
        return(self.valuetuple() == other.valuetuple())

    def __add__(self, other):
        n = self._inside_len()
        if n != other._inside_len():
            raise(KeyError("Cannot add ParamDict of len {} to one of len {}".format(
                other._inside_len(), n)))
        newD = copy.deepcopy(self)
        for i in range(n):
            newD[i] = self[i] + other[i]
        return(newD)

    def __mul__(self, other):
        """Other must be a scalar type"""
        newD = copy.deepcopy(self)
        for key in self:
            if isinstance(self[key], (list, np.ndarray)):
                for i in range(len(self[key])):
                    newD[key][i] = self[key][i] * other
            else:
                newD[key] = other * self[key]
        return(newD)

    def __rmul__(self, other):
        return(self.__mul__(other))

    def __sub__(self, other):
        return(self + (-1 * other))
