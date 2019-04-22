import networkx as nx
import copy as cp
import cliquergm.statistic as statistic
import numpy as np
from cliquergm.plotstyles import plot_imports
import cliquergm.sample_tools as st

"""Class implementing undirected graphs without self loops

Nodes are identified by number. Self loops and multiple edges are
not supported.
"""


# ######## Graph Generator Functions ########
def empty_graph(n):
    """Return an empty graph with n nodes."""
    G = Graph(n)
    return (G)


def complete_graph(n):
    """Return a complete graph with n nodes."""
    G = Graph(n)
    G.make_complete()
    return (G)


def erdos_renyi_graph(n, p):
    """Return an Erdos-Renyi graph with n nodes.

    Parameters
    ----------
    n : int
        Positive number of nodes

    p : float
        Target density of graph between 0 and 1

    Returns
    -------
    G : Graph
    """
    G = Graph(n)
    G.make_complete()
    edgelist = G.edgelist()
    for edge in edgelist:
        if np.random.rand() > p:
            G.remove_edge(*edge)
    return (G)


def import_nx_graph(g):
    """Allows conversion of a networkx graph object."""
    newG = Graph(g.number_of_nodes())
    for edge in g.edges():
        newG.add_edge(*edge)
    return(newG)


class Graph(object):
    """Class for undirected graphs without self loops or multiple edges.

    Attributes
    ----------
    number_of_nodes : int

    graph : dict
        Adjacency list stored in dictionary of sets
        keyed by node name (integers by default).

    stats : dict
        (default ``None``)
        Initialized by add_stats method. Dictionary keyed by subclass names
        corresponding to subgraph statistics of interest. Values are statistic
        objects.

    iteration : int
        (default ``0``)
        Graph iteration number, incremented when an edge is added or removed
        during sampling.

    last_edge : tuple
        (default ``None``)
        Last edge whose state was modified during sampling. Functions which
        change more than one edge in the graph should set last_edge to None.

    last_edge_added : boolean
        (default ``None``)
        Records whether last_edge was added or removed.

    alledges : list
        List of all possible edge tuples, without duplication.

    Notes
    -----
    Nodes are keyed by number and node names should not be changed. Self loops
    and multiple edges are not allowed.

    A graph object keeps track of the most recently added or removed
    edge, so that graph state can be reverted easily to before the last
    change. Statistic counts are also reverted.

    Direct computation of difficult subgraph counts (such as cliques,
    triangles) is handled by the networkx package, and methods are implemented
    in this class. Methods for efficient, progressive calculation of statistic
    counts are implemented in each statistic subclass.
    """

    def __init__(self, n):
        """Initialize a graph with n nodes"""
        self.number_of_nodes = n
        self.graph = {i: set() for i in range(n)}

        self.stats = None
        self.iteration = 0

        self.last_edge = None
        self.last_edge_added = False
        self.alledges = list({frozenset({i, j}) for i in range(n)
                              for j in range(n) if i != j})
        self.ergm_P = None  # This will hold the ERGM exponent, not the scaled
        # probability.
        # Choice good for a default value? Initialized for sampling by
        # add_stats method.

    def copy(self):
        """Return an exact copy of a graph object.

        Copy function makes an exact copy of the graph object, including all
        statistics in the self.stats dictionary. This is faster than
        copy.deepcopy() and assures that each statistic's graph pointer is
        manually reassigned to the new graph (very important).
        """
        Gnew = Graph(self.number_of_nodes)
        Gnew.graph = cp.deepcopy(self.graph)

        if self.stats is not None:
            Gnew.stats = {key: self.stats[key].copy()
                          for key in self.stats.keys()}
            # New statistic objects have graph attributes pointing to self.
            # Reassign them to point to Gnew:
            for stat in Gnew.stats.values():
                stat.graph = Gnew
        else:
            Gnew.stats = None
        Gnew.iteration = self.iteration

        Gnew.last_edge = self.last_edge
        Gnew.last_edge_added = self.last_edge_added
        Gnew.alledges = self.alledges.copy()
        Gnew.ergm_P = self.ergm_P
        return(Gnew)

    # ######## Graph Modifying Functions (User) ########
    def iterate(self):
        """ Iterate current graph once.

        Iterate function proposes adding or removing random edges to/from
        current_graph until one proposal is accepted based on graph
        Statistic parameters, using Metropolis-Hastings acceptance
        criteria [1]_.

        References
        ----------
        ..  [1] Robert, C. P. G., Casella (2004): Monte Carlo Statistical
            Methods. Chapter 7. <https://doi.org/10.1007/978-1-4757-4145-2>
        """

        if np.random.rand() < 0.02:
            self.invert()
            self.update_stat_counts()
            new_P = self.probability()
            r = _exp(new_P - self.ergm_P)  # This is a ratio of ERGM probabs
            if np.random.rand() >= r:
                self.invert()
                for stat in self.stats.values():
                    stat.revert()
            else:
                self.ergm_P = new_P
        else:
            while True:  # Choose valid edge
                n1, n2 = np.random.randint(0, self.number_of_nodes, 2)
                if n1 != n2:
                    break
            # Create proposal graph
            self.change_edge(n1, n2)

            # Decide whether to keep change using relevant counts
            self.update_stat_counts()
            new_P = self.probability()
            r = _exp(new_P - self.ergm_P)  # This is a ratio of ERGM probabs
            # If r >= 1, new graph will always be accepted.
            if np.random.rand() >= r:
                # If change rejected, revert graph state
                self.revert()
            else:
                self.ergm_P = new_P
        self.iteration += 1
        return

    def change_edge(self, n1, n2):
        """Add edge if edge does not exist, remove if it does."""
        if not self.has_edge(n1, n2):
            self.add_edge(n1, n2)
        else:
            self.remove_edge(n1, n2)

    def add_edge(self, n1, n2):
        """Add edge ``(n1, n2)`` to the graph.

        No indication will be given if edge ``(n1, n2)`` already exists.
        """
        # Should I exclude self loops? Also, if nodes are passed that are not
        # in the graph, one may be added to an adjacency set.
        if n1 == n2:
            print("Debug warning: Attempt to add self loop. Graph unchanged.")
            return
        self.graph[n1].add(n2)
        self.graph[n2].add(n1)
        self.last_edge = (n1, n2)
        self.last_edge_added = True

    def remove_edge(self, n1, n2):
        """Remove edge ``(n1, n2)`` from the graph."""
        # Fastest way to remove an edge as long as edge exists, but slower than
        # set subtraction if edge doesn't exist. Avoid calling remove_edge on an
        # edge that may not exist. TODO: Is this the right choice?
        # Note: Last edge only updated if change is made!
        try:
            self.graph[n1].remove(n2)
            self.graph[n2].remove(n1)
            self.last_edge = (n1, n2)
            self.last_edge_added = False
        except KeyError:
            print("Graph has no edge ({}, {})".format(n1, n2))

    def revert(self):
        """Undo last edge addition or subtraction.

        Repeated calls of revert method have the same effect as calling revert
        once.
        """
        if self.last_edge:  # Only try to revert if a last change is recorded
            if self.last_edge_added:
                self.remove_edge(*self.last_edge)
            else:
                self.add_edge(*self.last_edge)
            if self.stats:
                for stat in self.stats.values():
                    stat.revert()
            self.last_edge = None  # Is this the right choice?
        else:
            raise UserWarning("Graph.revert() called twice in a row, or on",
                              "unmodified graph object. No change made.")

    def invert(self):
        """Change the state of each edge in the graph."""
        for edge in self.alledges:
            self.change_edge(*edge)
        self.last_edge = None

    def update_stat_counts(self):
        """Calculate statistic subgraph counts of interest.

        Notes
        -----
        Assumes statistic count attributes describe the last graph, and that
        last_edge and last_edge_added are recently updated by some
        change to the graph. Run once after changing the graph.
        TODO: calling update_stat_counts multiple times in a row may result
        in inaccurate counts. See edge.update_count function.
        """
        for stat in self.stats.values():
            stat.update_count()

    def add_stats(self, stats):
        """Add statistics of interest to the graph, and calculate counts.

        Function will overwrite any existing statistics. If None is passed then
        no action will be taken.

        Parameters
        ----------
        stats : dictionary
            Dictionary keyed by subclass names corresponding to subgraph
            statistics of interest. Values are statistic parameters.
        """
        if stats is None:
            return
        self.stats = {key:  statistic.names[key](
            stats[key], self) for key in stats.keys()}
        self.update_stat_counts()
        self.ergm_P = self.probability()

    def _update_parameter_vect(self, v):
        """Updates statistic parameters from a vector of precise length and
        order.

        The order must match the order of the keys of
        cliquergm.statistic.names."""
        for key in statistic.names:  # order important, use statistic.names.
            if key in self.stats:
                if isinstance(self.stats[key].parameter, list):
                    ind = len(self.stats[key].parameter)
                    self.stats[key].update_parameter(v[:ind])
                    v = v[ind:]
                else:
                    self.stats[key].update_parameter(v[0])
                    v = v[1:]

    def change_parameter(self, stats):
        """Update statistic parameters.

        Will raise KeyError if stats contains keys for statistics not already
        in graph stats dictionary. To add new statistics to the graph, pass a
        dictionary containing all statistics of interest to the add_stats()
        method.
        """
        for key in stats.keys():
            self.stats[key].update_parameter(stats[key])

    def make_nx_graph(self):
        """Return networkx graph object with edges from graph."""
        g = nx.empty_graph(self.number_of_nodes)
        g.add_edges_from(self.edgelist())
        return(g)

    def return_neighbors(self):
        """Return all the neighbors of the current graph in a list

        Iterates over self.edgelist(), changing the state of each in a returned
        graph """
        gcomplete = complete_graph(self.number_of_nodes)
        edges = gcomplete.edgelist()
        return_arr = []

        for edge in edges():
            newG = self.copy()
            newG.change_edge(*edge)
            newG.update_stat_counts()
        return(return_arr)

    # ######## Graph State Functions ########
    def probability(self):
        r"""Return exponent representing the scaled log-probability of the
        graph.

        Calculates the probability of the graph's current configuration
        based on the current values in Statistics' counts. The probability of
        a graph configuration is given by

        .. math::
            P(G) = \frac{1}{c} \exp\{\sigma_1 S_1(G)
            + \sigma_2 S_2(G) + \ldots\}


        where G is the graph configuration, c is an unknown constant,
        :math:`\sigma_i` denotes the |ith| statistic parameter, and
        :math:`S_i(G)` denotes the |ith| statistic subgraph count in G [1]_.

        This method only returns the exponent

        .. math::
            \sigma_1 S_1(G) + \sigma_2 S_2(G) + \ldots

        .. |ith| replace:: i\ :sup:`th`

        References
        ----------
        ..  [1] Robins, G., Pattison, P., Kalish, Y., & Lusher, D. (2007). An
            introduction to exponential random graph (p*) models for social
            networks. *Social networks*, 29(2), 173-191.
            <https://doi.org/10.1016/j.socnet.2006.08.002>
        """
        exP = 0
        for stat in self.stats.values():
            exP += np.dot(stat.count, stat.parameter)
        return(exP)

    def has_edge(self, n1, n2):
        return (n1 in self.graph[n2])

    def edgelist(self):
        elist = []
        for node, nodeset in self.graph.items():
            for tnode in nodeset:
                if (node, tnode) not in elist and (tnode, node) not in elist:
                    elist.append((node, tnode))
                # Could also do this test by restricting tnode value to one
                # greater than node, when node > 1.
        return(elist)

    def number_of_edges(self):
        edges = 0
        for nodeset in self.graph.values():
            edges += len(nodeset)
        edges = int(edges / 2)
        return(edges)

    def counts(self):
        """Return numpy array summarizing statistic subgraph counts.

        Returns flat numpy array of statistic counts (including counts of each
        size of subgraphs with variable size), with consistent order based on
        order of statistics dictionary.
        """
        return_array = np.array([], dtype=int)
        for key in statistic.names:  # order important, use statistic.names.
            if key in self.stats:
                return_array = np.append(return_array, self.stats[key].count)
        return(return_array)

    def parameters(self):
        """Return numpy array summarizing statistic parameters.

        Concatenates statistic parameters in a single vector using the
        order of statistic.names."""
        return_array = np.array([], dtype=float)
        for key in statistic.names:  # order important, use statistic.names.
            if key in self.stats:
                return_array = np.append(return_array,
                                         self.stats[key].parameter)
        return(return_array)

    def count_dict(self, split=False):
        """Return dictionary summarizing statistic subgraph counts.

        Returns dictionary of statistic counts keyed by statistic names. If
        split=True, function splits statistic counts that are arrays,
        so that the ith element of the count array will be keyed by
        ``"<i+1>-<statistic name>"``.
        """
        return_dict = {}
        for key in statistic.names:  # order important, use statistic.names.
            if key in self.stats:
                ct = self.stats[key].count
                if not split:
                    return_dict.update({key: ct})
                elif isinstance(ct, int):
                    return_dict.update({key: np.array(ct)})
                elif isinstance(ct, list):
                    for i in range(len(ct)):
                        ikey = str(i+1) + "-" + key
                        return_dict.update({ikey: np.array(ct[i])})
        if not split:
            return(st.ParamDict(return_dict))
        else:
            return(return_dict)

    def parameter_dict(self):
        return_dict = {key: self.stats[key].parameter for key in self.stats}
        return(st.ParamDict(return_dict))

    # ## Graph State Functions -- Subgraph counters and other statistics ##
    def find_cliques(self):
        """Use networkx package to list maximal clique node membership.

        Returns
        -------
        cliques : generator object
            Generator containing lists of maximal clique node membership in
            the graph.
        """
        g = self.make_nx_graph()
        return (nx.find_cliques(g))

    def triangles(self):
        """Use networkx package to count and return number of triangles."""
        g = self.make_nx_graph()
        return ((int)(sum(nx.triangles(g).values()) / 3))

    def density(self):
        """Return edge density of the graph."""
        n = self.number_of_nodes
        density = 2 * self.number_of_edges() / (n * (n - 1))
        return(density)

    def diameter(self):
        """Use networkx package to calculate and return graph diameter."""
        return (nx.diameter(self.make_nx_graph()))

    def degree_histogram(self):
        """Use networkx package to calculate and return a degree histogram.
        TODO: Rewrite to not use nx (very easy)

        Returns
        -------
        histogram : array
            Array whose |ith| element is the count of nodes with degree i.
        """
        return (nx.degree_histogram(self.make_nx_graph()))

    def is_connected(self):
        """Use networkx package to determine whether graph is connected."""
        return (nx.is_connected(self.make_nx_graph()))

    # ######## Modifier functions for use by graph generators ########
    def make_empty(self):
        """Remove all edges from the graph."""
        self.graph = {i: set() for i in range(self.number_of_nodes)}
        self.last_edge = None

    def make_complete(self):
        """Add all possible edges to the graph."""
        self.graph = {
            i: set(range(self.number_of_nodes)) - {i}
            for i in range(self.number_of_nodes)
        }
        self.last_edge = None

    # ######## Printing functions for use by chain sampling methods ########
    def print_intro(self):
        """Print formatted table header with each graph parameter label."""
        string = "\nIteration"
        for key in statistic.names:
            if key in self.stats:
                string += "|\t{}\t".format(self.stats[key].print_label())
        line_length = len(string) - 2 + 6 * string.count("\t")
        string += ("\n\n---------|"
                   + "".join(["-" for n in range(line_length - 10)]) + "\n")
        return(string)

    def print_summary(self):
        """Print formatted list of statistic subgraph counts."""
        string = str(self.iteration) + "\t "
        for key in statistic.names:
            if key in self.stats:
                string += "|\t{}\t".format(self.stats[key].print_count())
        string += "\n"
        return(string)

    def print_debug_summary(self):
        """Print formatted list of statistic subgraph counts.

        Uses a verified method to calculate subgraph counts.
        """
        string = str(self.iteration) + "\t "
        for key in statistic.names:
            if key in self.stats:
                string += "|\t{}\t".format(self.stats[key].debug_count())
        string += "\n"
        return(string)

    # ######## Graphics Functions ########
    def layout(self, dim=2, algorithm="kk"):
        """ Calculate graph layout, return as dictionary.

        Parameters
        ----------
        algorithm : string
            (default ``"kk"``)
            Algorithm used to compute graph layout. Choose from:
            ``"kk"`` (Kamada Kawai), ``"circular"``, ``"random"``,
            ``"spring"``, ``"spectral"``, or ``"cliques"``.
        """
        lout = {"kk":       nx.drawing.layout.kamada_kawai_layout,
                "circular": nx.drawing.layout.circular_layout,
                "random":   nx.drawing.layout.random_layout,
                "spring":   nx.drawing.layout.spring_layout,
                "spectral": nx.drawing.layout.spectral_layout,
                "cliques":  _clique_groups}
        G = self.make_nx_graph()
        pos = lout[algorithm](G, dim=dim)
        return(pos)

    @plot_imports
    def plot(self, show=True, algorithm="cliques", save=None, figures=None, modules=None,
             **kwargs):
        """ Open a graph plot.

        Parameters
        ----------
        show : bool
            (Default ``True``)
            Whether to show the plots produced by this method. Figures can be
            silently appended to the list passed to parameter ``figures`` by
            specifying a value of ``False``.

        algorithm : string
            (Default ``"cliques"``)
            Algorithm used by graph.layout() method to compute graph layout.
            For available options, see :meth:`~cliquergm.graph.Graph.layout()`.

        save : string
            (Default ``None``)
            File format for export. Choose from: ``"png"``, ``"svg"``,
            ``"jpeg"``, ``"eps"``, and ``"pdf"``. Image will be saved as
            ``"graph.<save>"`` in the default browser's Downloads folder. For
            more control over all aspects of the figure, such as image size,
            use output variable 'figures', or pass options to
            plotly.offline.plot() function as keyword arguments.

        figures : array
            (Default ``None``)
            If passed, the resulting plotly.graph_objs.Figure object will be
            appended to figures.
        """
        pl, ps, *_ = modules
        imgsize = kwargs.pop("imgsize")
        pos = self.layout(algorithm=algorithm)
        nodexytext = [[],[],[]]
        edgexy = [[],[]]
        for node in pos:
            nodexytext[0].append(pos[node][0])
            nodexytext[1].append(pos[node][1])
            nodexytext[2].append("Node degree: " + str(len(self.graph[node])))
        nodes = ps.nodescatter(*nodexytext)
        for edge in self.edgelist():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edgexy[0].extend([x0, x1, None])
            edgexy[1].extend([y0, y1, None])
        edges = ps.edgescatter(*edgexy)
        fig = pl.graph_objs.Figure(data=[edges, nodes],
                           layout=ps.graphlayout())
        if show:
            pl.offline.plot(fig, filename="graph.html", **kwargs)
        if save:
            pl.io.write_image(fig, "graph.pdf", width=imgsize[0], height=imgsize[1])
        if figures is not None:
            figures.append(fig)

    @plot_imports
    def plot3d(self, algorithm="random", save=None, figures=None, modules=None,
               **kwargs):
        """ Open a 3-dimensional graph plot.

        Parameters
        ----------
        algorithm : string
            (default "random")
            Algorithm used by graph.layout() method to compute graph layout.
            Only "random" and "spectral" result in varied z-coordinates.

        save : string
            (default None)
            File format for export. Choose from: ``"png"``, ``"svg"``,
            ``"jpeg"``, ``"eps"``, and ``"pdf"``. Image will be saved as
            ``"graph.<save>"`` in the default browser's Downloads folder. For
            more control over all aspects of the figure, such as image size,
            utilize output variable figures.

        figures : array
            (default ``None``)
            If passed, the resulting plotly.graph_objs.Figure object will be
            appended to figures.
        """
        pl, ps = modules
        pos = self.layout(algorithm=algorithm, dim=3)
        edges = ps.edgescatter3d()
        nodes = ps.nodescatter3d()
        for node in pos:
            nodes.x.append(pos[node][0])
            nodes.y.append(pos[node][1])
            nodes.z.append(pos[node][2])
            nodes.text.append("Node degree: " + str(len(self.graph[node])))
        for edge in self.edgelist():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edges.x.extend([x0, x1, None])
            edges.y.extend([y0, y1, None])
            edges.z.extend([z0, z1, None])
        fig = pl.graph_objs.Figure(data=[edges, nodes],
                           layout=ps.graphlayout3d())
        pl.offline.plot(fig, filename="graph.html",
                 image=save, image_filename="graph", **kwargs)
        if figures is not None:
            figures.append(fig)


@plot_imports
def plot(G, pos, save=None, figures=None, modules=None, **kwargs):
    pl, ps = modules
    edges = ps.edgescatter()
    nodes = ps.nodescatter()
    for node in pos:
        nodes.x.append(pos[node][0])
        nodes.y.append(pos[node][1])
        nodes.text.append("Node degree: " + str(len(G.graph[node])))
    for edge in G.edgelist():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edges.x.extend([x0, x1, None])
        edges.y.extend([y0, y1, None])
    fig = pl.graph_objs.Figure(data=[edges, nodes],
                       layout=ps.graphlayout())
    pl.offline.plot(fig, filename="graph.html",
             image=save, image_filename="graph", **kwargs)
    if figures is not None:
        figures.append(fig)


def _exp(expon):
    """ A wrapper for numpy exp which evaluates to 1 for positive exponents.
    """
    if expon >= 0:
        return(1.0)
    else:
        return(np.exp(expon))


def _rotate(vec, radians):
    c, s = np.cos(radians), np.sin(radians)
    R = np.array([[c, -s], [s, c]])
    return(np.matmul(R, vec))


def _clique_groups(G, dim=2):
    numnodes = G.number_of_nodes()
    cliquelists = list(nx.find_cliques(G))
    cliquelists.sort(key=len, reverse=True)
    nodessofar = set()
    enoughcliques = []
    while len(nodessofar) < numnodes:
        intersectionlist = [len(nodessofar.union(set(cliquelist)))
                            for cliquelist in cliquelists]
        i = intersectionlist.index(max(intersectionlist))
        nodessofar = nodessofar.union(set(cliquelists[i]))
        enoughcliques.append(cliquelists[i])

    n = len(enoughcliques)
    cliquecenters = nx.drawing.layout.circular_layout(
        list(range(n)), center=(0, 0), scale=(2.4*n / 6.28))
    enoughcliques.sort(key=len)
    pos = {}
    for i in range(n):
        cpos = nx.drawing.layout.circular_layout(
            enoughcliques[i], center=(0,0), scale=1)
        for key in cpos:
            cpos[key] = _rotate(cpos[key], 2 * i * np.pi / n) + cliquecenters[i]
        pos.update(cpos)
    return(pos)


def _overlapping_clique_groups(G, dim=2):
    """An idea that should be implemented"""
    numnodes = G.number_of_nodes()
    print(numnodes)
    cliquelists = list(nx.find_cliques(G))
    cliquelists.sort(key=len, reverse=True)
    nodessofar = set()
    enoughcliques = []
    while len(nodessofar) < numnodes:
        print(len(nodessofar))
        intersectionlist = [len(nodessofar.union(set(cliquelist)))
                            for cliquelist in cliquelists]
        print(intersectionlist)
        i = intersectionlist.index(max(intersectionlist))
        print(cliquelists[i])
        nodessofar = nodessofar.union(set(cliquelists[i]))
        enoughcliques.append(cliquelists[i])
    enoughcliques.sort(key=len, reverse=True)
    mergedcliques = []
    n = len(enoughcliques)
    print(n)
    print(enoughcliques)
    cliquecenters = nx.drawing.layout.circular_layout(
        list(range(n)), center=(0, 0), scale=(2.4*n / 6.28))
    print(cliquecenters)
    enoughcliques.sort(key=len)
    print(enoughcliques)
    pos = {}
    for i in range(n):
        cpos = nx.drawing.layout.circular_layout(
            enoughcliques[i], center=cliquecenters[i], scale=1)
        pos.update(cpos)
    return(pos)
