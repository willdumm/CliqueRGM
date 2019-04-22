from cliquergm.model import Cmodel
import numpy as np

def test_edge_param_works():
    np.random.seed(2)
    ips1 = {'Cliques': [0], 'Edge': 0}
    ips2 = {'Cliques': [0], 'Edge': 1}
    ips3 = {'Cliques': [0], 'Edge': -1}
    model = Cmodel(10, [ips1, ips2, ips3])
    model.sample_converged(100)
    cdictlist = [chain.avg_count_dict() for chain in model.chains]
    edgenumlist = [dic['Edge'] for dic in cdictlist]
    warnstrings = ["Edge parameter 0 doesn't yield expected edge count",
                   "Edge parameter 1 doesn't yield expected edge count",
                   "Edge parameter -1 doesn't yield expected edge count"]
    assert edgenumlist[0] <= 25 and edgenumlist[0] >= 20, warnstrings[0]
    assert edgenumlist[1] <= 35 and edgenumlist[1] >= 30.5, warnstrings[1]
    assert edgenumlist[2] <= 15 and edgenumlist[2] >= 10, warnstrings[2]
