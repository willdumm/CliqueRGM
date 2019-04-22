from cliquergm.model import fit
import numpy as np


def test_fit_with_bounds():
    bounds = {'Cliques': [(0, 0), (-2, 2)], 'Edge': None}
    target_counts = {'Cliques': [0, 1], 'Edge': 1}
    np.random.seed(0)
    fit(target_counts, 2, 0.3, 0.2, bounds=bounds, random_samples=2)


def test_fit_with_ips():
    stats = {'Cliques': [0, 0], 'Edge': 0}
    target_counts = {'Cliques': [0, 1], 'Edge': 1}
    np.random.seed(22)
    fit(target_counts, 2, 0.3, 0.3, ips=stats)


def test_fit_with_ips_bounds():
    stats = {'Cliques': [0, 0], 'Edge': 0}
    bounds = {'Cliques': [(0, 0), (-2, 2)], 'Edge': None}
    target_counts = {'Cliques': [0, 1], 'Edge': 1}
    np.random.seed(20)
    fit(target_counts, 2, 0.3, 0.3, ips=stats, bounds=bounds)
