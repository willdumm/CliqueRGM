from . import sample_tools as st
import copy
import collections


class Statistic(object):
    """
    Attributes
    ----------

    parameter :
        model parameter associated with the Statistic subgraph
    count :
        current count of Statistic subgraph in self.graph
    last_count :
        most recent count, stored for revert method
    graph :
        graph object to which Statistic object belongs

    Notes
    -----

    The Statistic class provides basic structure and methods for a Statistic
    object. Specific Statistics can be implemented as subclasses, and must
    implement the following methods:

    update_count(self) :
        Calculates count of Statistic subgraph in self.graph using an efficient
        method (employ self.graph.last_edge and last_edge_added) and store
        result in self.count. Update self.last_count.

    robust_count(self) :
        Calculates count of Statistic subgraph using a robust method known to
        work.

    If Statistic is nonstandard (i.e. requires extra attributes or has count
    in array form) additional methods will need to be overriden, such as
    __init__, copy, print_label, print_count, max_len, and revert.
    """
    _count_type = int

    def __init__(self, parameter, graph):
        """Initialize a new statistic object.

        Parameters
        ----------
        parameter : float, or array of floats
            Model parameter associated with the subgraph statistic being
            initialized.

        graph : cliquergm.graph.Graph
            A pointer to the graph to which the parameter will belong.
        """
        self.parameter = parameter
        self.graph = graph

        self.count = None
        self.last_count = None

    def copy(self):
        """Return an exact copy of the statistic object.

        self.graph pointer remains unchanged. If copy is called by the copy
        method for graph, that method must manually reassign the graph pointer
        to the new graph object.
        """
        newstat = self.__class__(self.parameter, self.graph)
        newstat.count = self.count
        newstat.last_count = self.last_count
        return(newstat)

    def update_parameter(self, parameter):
        """Reassign the statistic parameter."""
        self.parameter = parameter

    def revert(self):
        """Revert statistic to the state before last call to update_count()."""
        self.count = self.last_count
        self.last_count = None

    def print_label(self):
        """Return the first six letters of the statistic name as a string.

        Accommodates statistics which are of variable size, and labels
        columns starting from 1 for each element of the count array.
        """
        if isinstance(self.count, int):
            return(self.__class__.__name__[0:6])
        elif isinstance(self.count, list):
            return("size\t"
                   + "\t".join([str(x + 1) for x in range(len(self.count))])
                   + "  " + self.__class__.__name__[0:6])
        else:
            print(type(self.count))
            print("\n A custom print_label() method must be implemented for",
                  "the statistic subclass", self.__class__.__name__)
            exit()

    def print_count(self):
        """Return the current count of the statistic subgraph as a string

        Accommodates statistic counts which are integers and arrays of
        integers.
        """
        if isinstance(self.count, int):
            return(str(self.count))
        elif isinstance(self.count, list):
            return("    \t"
                   + "\t".join([str(x) for x in st.pad_zeros(self.count)])
                   + "        ")
        else:
            print("A custom print_count() method must be implemented for the",
                  "statistic subclass", self.__class__.__name__)
            exit()

    def debug_count(self):
        """Return a formatted statistic count calculated with a robust method.
        """
        count = self.robust_count()
        if isinstance(count, int):
            return(str(count))
        elif isinstance(count, list):
            return("    \t" + "\t".join([str(x) for x in st.pad_zeros(count)])
                   + "        ")
        else:
            print("A custom debug_count() method must be implemented for the",
                  "statistic subclass", self.__class__.__name__)
            exit()

    def update_count(self):
        """Infer the current count of the statistic subgraph in the graph.

        update_count is implemented in each statistic subclass.
        """
        print("A custom update_count method must be implemented for the",
              "statistic subclass", self.__class__.__name__)
        return

    def robust_count(self):
        """Calculate the statistic subgraph count using a trusted method.

        robust_count is implemented in each statistic subclass.
        """
        print("A custom robust_count method must be implemented for the",
              "statistic subclass", self.__class__.__name__)
        return


class Cliques(Statistic):
    _count_type = list

    def __init__(self, parameter, graph):
        super().__init__(parameter, graph)

        self._clique_set = None
        self._last_clique_set = None
        # Extend/trim given parameter list to match size of graph
        self.parameter.extend([0] * (graph.number_of_nodes - len(parameter)))
        self.parameter = parameter[0: graph.number_of_nodes]

    def max_len(graph_size):
        """Return the maximum length of the
        parameter list, as a function of graph size"""
        return(graph_size)

    def copy(self):
        newstat = super().copy()
        newstat._clique_set = copy.copy(self._clique_set)
        newstat._last_clique_set = copy.copy(self._last_clique_set)
        return(newstat)

    def revert(self):
        super().revert()
        self._clique_set = self._last_clique_set
        self._last_clique_set = None

    def update_count(self):
        self.last_count = self.count
        self._last_clique_set = self._clique_set
        self._clique_set = self._count_clique_set()
        self.count = st.dist(self._clique_set, self.graph.number_of_nodes)

    def update_parameter(self, parameter):
        super().update_parameter(parameter)
        self.parameter = list(self.parameter)

    def robust_count(self):
        return(st.dist(self.graph.find_cliques(), self.graph.number_of_nodes))

    def _count_clique_set(self):
        """ Infer the clique set of G.

        Assumes that the statistic attribute '_cliqueset' describes cliques in
        the previous graph configuration. Considers the fact that only the
        supplied edge is changed. If edge is not supplied, uses nx.find_cliques
        to return set of frozen sets of clique members in graph. Does NOT
        update statistic attributes.
        """
        if self.graph.last_edge is None or self._clique_set is None:
            # print("Debug warning: using slow method to count cliques!")
            return(set(map(frozenset, self.graph.find_cliques())))
        else:
            x, y = self.graph.last_edge
            cliqueset = copy.copy(self._clique_set)
            # Since frozen sets are immutable, shallow copy is enough. However,
            # copy() is necessary to avoid changing graph attribute.

            Fx = {c for c in cliqueset if x in c}
            Fy = {c for c in cliqueset if y in c}

            if self.graph.last_edge_added:  # Check an edge that was added to G
                F = Fx.union(Fy)
                Nx = set()

                for Ci in Fx:
                    Nxi = set({y})
                    for u in Ci:
                        if self.graph.has_edge(y, u):
                            Nxi.add(u)
                    Nx.add(frozenset(Nxi))

                Nmaximal = {C for C in Nx if not
                            st.ispropersubset_of_element(C, Nx)}
                Q = Nmaximal | {C for C in F if not
                                st.ispropersubset_of_element(C, Nmaximal)}
                cliqueset = cliqueset - F
                cliqueset.update(Q)

            else:                       # Check an edge that was removed from G
                Fxy = Fx.intersection(Fy)
                # Remove maximal cliques in G that relied on edge (x,y)
                cliqueset = cliqueset - Fxy
                Fxonly = Fx - Fxy
                Fyonly = Fy - Fxy
                for clique in Fxy:
                    Cx = frozenset(clique - {y})
                    Cy = frozenset(clique - {x})

                    # Add Cx if there doesn't exist M in Fx' such that Cx is
                    # a subset of M. Fx' = Fx - Fxy, is Fx of new graph.
                    if not st.issubset_of_element(Cx, Fxonly):
                        cliqueset.add(Cx)

                    # Add Cy if there doesn't exist M in Fy' such that Cy is
                    # a subset of M. Fy' = Fy - Fxy, is Fy of new graph.
                    if not st.issubset_of_element(Cy, Fyonly):
                        cliqueset.add(Cy)
            return(cliqueset)


class Triangle(Statistic):
    def update_count(self):
        self.last_count = self.count
        self.count = self.graph.triangles()

    def robust_count(self):
        return(self.graph.triangles())


class Edge(Statistic):
    def update_count(self):
        self.last_count = self.count
        if self.graph.last_edge is None or self.count is None:
            self.count = self.graph.number_of_edges()
        else:
            # TODO if update_count is called multiple times in a row without
            # changes made to graph, this will continue to add or subtract
            # edges from the graph. Bad behavior.
            self.count = self.count + (-1) + self.graph.last_edge_added * (2)

    def robust_count(self):
        return(self.graph.number_of_edges())


# Each other module may be using a distinct names variable (I think a new,
# independent one is created by each 'import statistic' statement). However,
# Statistic.__subclasses__() returns an array of subclasses, in the order they
# appear in the file, so order should be preserved between all files.
# This will be used to order count summaries and printing.

names = collections.OrderedDict({stat.__name__: stat
                                 for stat in Statistic.__subclasses__()})

## Needs modification, and may never be needed at all
# def tup_to_pars(ls):
#     i = 0  # ls Array index
#     output = copy.deepcopy(ips)
#     for key in names:
#         if key in output:
#             if isinstance(output[key], list):
#                 for j in range(len(ips[key])):
#                     output[key][j] = ls[i]
#                     i = i + 1
#             else:
#                 output[key] = ls[i]
#                 i = i+1
#     return(output)


def dict_ind():
    raise(NotImplementedError("This is deprecated. Need to be using ParamDict class"))


def pars_to_tup(pars):
    raise(NotImplementedError("This is deprecated. Need to be using ParamDict class"))


def keytuple(d):
    raise(NotImplementedError("This is deprecated. Need to be using ParamDict class"))
