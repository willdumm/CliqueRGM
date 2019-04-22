from cliquergm import graph as gr
import numpy as np
from cliquergm import sample_tools as st
from cliquergm import model as m
from cliquergm import statistic as stat


def test_clique_counts():
    # Tested
    np.random.seed(seed=1)
    G = gr.erdos_renyi_graph(10, 0.5)
    stats = {'Cliques': [0] * 10}
    G.add_stats(stats)
    for i in range(1000):
        while True:  # Choose valid edge
            n1 = np.random.randint(0, G.number_of_nodes)
            n2 = np.random.randint(0, G.number_of_nodes)
            if n1 != n2:
                break
        G.change_edge(n1, n2)
        G.update_stat_counts()
        assert(G.stats['Cliques'].robust_count() == G.stats['Cliques'].count)


def test_edge_counts():
    np.random.seed(seed=1)
    G = gr.erdos_renyi_graph(10, 0.5)
    stats = {'Edge': 0}
    G.add_stats(stats)
    for i in range(1000):
        while True:  # Choose valid edge
            n1 = np.random.randint(0, G.number_of_nodes)
            n2 = np.random.randint(0, G.number_of_nodes)
            if n1 != n2:
                break
        G.change_edge(n1, n2)
        G.update_stat_counts()
        assert(G.stats['Edge'].robust_count() == G.stats['Edge'].count)


def test_counts_in_sampler():
    np.random.seed(seed=1)
    # Test counts of all subgraphs, and iteration number.
    # These stats cannot be all zero, because then no edges are ever rejected.
    stats = {key: stat.names[key]._count_type() for key in stat.names}
    stats['Cliques'] = [1, .1, -.5, .1, -.1, .1, -.05]
    model = m.Model([gr.empty_graph(10)], stats)
    model.update_prefs(output="None", burnin=0, sample_interval=2)
    model.sample(500)
    for key in stats:
        flag = True
        for i in range(500):
            g = model.chains[0].sample_list[i]
            # print(g.stats[key].count, g.stats[key].robust_count())
            if g.stats[key].count != g.stats[key].robust_count():
                flag = False
                break
        assert flag, "Counts appear wrong for {}".format(key)
    warnstring = "Iteration logic seems wrong"
    assert model.chains[0].sample_list[-1].iteration == 1000, warnstring


def test_erdos_renyi_property():
    np.random.seed(seed=1)
    stats = {'Cliques': [0], 'Edge': 0}
    er_graphs = []
    for _ in range(500):
        g1 = gr.erdos_renyi_graph(10, 0.5)
        g1.add_stats(stats)
        er_graphs.append(g1)
    model = m.Model([gr.empty_graph(10)], stats)
    model.update_prefs(output="None", burnin=500)
    model.sample(500)
    c1 = model.chains[0].avg_count_dict()
    c2 = st.avg_count_dict(er_graphs)
    d = st.avg_density(er_graphs)
    warnstring = ("This error may be ignored if these reasonably agree:"
                  + "\nMCMC density " + str(d) + " with average counts:\n"
                  + str(c1) + "\nErdos_renyi counts:\n" + str(c2)
                  + "\nDistance " + str(st.count_distance(c1, c2)))
    assert st.count_distance(c1, c2) < 2 and abs(.5-d) < .1, warnstring
