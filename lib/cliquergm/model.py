import multiprocessing as mp
from functools import partial
import cliquergm.chain as chain
import numpy as np
from cliquergm.plotstyles import plot_imports
from scipy.stats import wilcoxon, mannwhitneyu
import cliquergm.sample_tools as st
from copy import deepcopy
from cliquergm.statistic import names
import cliquergm.graph as gr
# import cliquergm.nullmodel as nm
import csv
import random
# Random only used for shuffling lists, nothing which determines
# sampling outcome.


"""Model object contains sample chains, sampling preferences, and methods."""


def _sample(chain, samples):
    # Used in concurrent sample method as wrapper for chain.sample method
    np.random.set_state(chain.randomstate)
    chainsampled = chain.sample(samples)
    chainsampled.randomstate = np.random.get_state()
    return(chainsampled)


def modelerr(target_counts, size, params, bbfailval=1000):
    """Samples from a converged ERGM using given parameters, and returns
    distance to the provided target subgraph counts.
    If an array of parameters is passed, then chains with each parameter set
    will be concurrently sampled, and an array of distances will be returned.
    If a single parameter set is passed, a single distance will be returned.
    """
    if isinstance(params, list):
        #TODO: make this function replace min_distance in the fit function.
        raise NotImplementedError
    else:
        params = list_to_pars(paramlist)
        model = Cmodel(size, [params])
        # model.update_prefs(output="Progress")  # For debug
        model.sample_converged(sample_period,
                               convergence_cutoff=convergence_cutoff,
                               concurrent=False)
        counts = model.chains[0].avg_count_dict()
        dist = st.count_distance(target_counts, counts, failval=bbfailval)
    return(dist)


def fit(target_counts, size, a, resolution, ips=None,
        bounds=None, random_samples=100, convergence_cutoff=6,
        sample_period=100, num_cores=4, min_dist=0.1, convergence_p=0.5,
        filename=None):
    """Seek parameter set which minimizes difference in subgraph counts from
    an observed subgraph count distribution

    Attributes
    ----------
    target_counts : dict
        Desired subgraph count distribution. A dictionary keyed by keys in
        statistic.names, with appropriate values for counts. Any target count
        not included in the target_count dictionary will be ignored.

    size : int
        Number of nodes in sampled graphs.

    a : float
        Initial parameter step value.

    resolution : float
        Minimum parameter step value. Defines the primary stopping criterion
        for the fitting algorithm.

    ips : dict
        Initial parameter guesses, in dictionary keyed by statistic.names.
        If provided, the fit algorithm will not perform the initial bounds
        search step. If no bounds are given, bounds corresponding to given
        initial parameters will be set to None, and corresponding to other
        parameters will be set to (0,0).

    bounds : dict
        Tuple-form inclusive bounds on parameters in ips, with same dictionary
        structure as ips. A None value for a bound will be interpreted as
        a bound of (-infty, infty), and a fixed parameter value t can be
        expressed as (t, t). If bounds are given, they must be given for all
        parameters for subgraphs of interest.

    random_samples : int
        The number of random parameter sets within parameter bounds to test
        at algorithm start.

    convergence_cutoff : int
        Passed to :meth:`Cmodel.sample_converged() <cliquergm.model.Cmodel.sample_converged()>`.

    sample_period : int
        The number of samples taken between each test for convergence of chains
        sampling from each proposed parameter set.

    num_cores : int
        Specifies how many steps in a distance-minimizing direction should be
        taken at once. Using a multiple of the total number of virtual CPU
        cores available will increase performance, but a decrease in num_cores
        will change how the fitting algorithm operates. Recommend
        `num_cores >= 4`.
    
    min_dist : float
        (Default ``0.1``)
        Specifies an alternative stopping criterion for the fitting algorithm.
        If the average distance between subgraph counts and target counts is
        less than ``min_dist``, the fitting algorithm will terminate, even if
        the minimum parameter step value passed as ``resolution`` has not
        yet been reached.

    convergence_p : float
        (Default ``0.05``)
        Specifies the p-value used to test convergence. A lower p-value makes
        it easier for convergence to be observed. A value of ``0`` has the
        effect that convergence of sampling chains is never tested, and the
        average subgraph counts observed in the first ``sample_period`` samples
        are used to assess parameter fit.

    filename : string
        (Default ``None``)
        If provided, information about each parameter vector tested by the
        fitting algorithm will be written to the file ``<filename>.csv`` upon
        completion of the fitting algorithm.

    Returns
    -------
    parameters : dict
        Optimal parameters found by the fitting algorithm

    sample_list : list
        A list of graphs sampled from the optimal parameter set.

    Notes
    -----
    If neither bounds nor parameters are passed,
    an exception will be raised. Initial parameters will be the parameters
    which the algorithm attempts to fit, and if None, the algorithm will
    attempt to fit parameters for which bounds are given.

    If bounds and initial parameters are provided, then bounds
    must be provided for every parameter in the initial parameter
    dictionary."""
    # Process input bounds and ips: This is horrible but it's actually quite a
    # complicated task. A naive approach is easiest to make work.
    if ips is None and bounds is None:
            raise TypeError("Provide either initial parameters or bounds")
    if bounds is None:
        bounds = {}
    # Bounds and ips if they exist should have all the same keys, including
    # all in target_counts.
    if ips is not None:
        ips.update({key: False for key in target_counts if key not in ips})
        ips.update({key: False for key in bounds if key not in ips})
    bounds.update({key: False for key in target_counts if key not in bounds})
    if ips is not None:
        bounds.update({key: False for key in ips if key not in bounds})
    # Fix bounds values that are False from previous step:
    if ips is None:
        for key in bounds:
            if bounds[key] is False:
                if names[key]._count_type is list:
                    bounds[key] = [(0, 0)]
                else:
                    bounds[key] = (0, 0)
    else:
        for key in bounds:
            if bounds[key] is False:
                if names[key]._count_type is list:
                    bounds[key] = []
                else:
                    if ips[key] is not False:
                        bounds[key] = None
                    else:
                        bounds[key] = (0, 0)
    # Extend bounds lists to full length
    for key in bounds:
        if names[key]._count_type is list:
            if ips is not None and ips[key] is not False:
                st.assert_len(bounds[key], len(ips[key]), None)
            st.assert_len(bounds[key], names[key].max_len(size), (0, 0))
    # Fix False values in ips and extend to full length
    if ips is not None:
        for key in ips:
            if ips[key] is False:
                if names[key]._count_type is list:
                    ips[key] = [0]
                else:
                    ips[key] = 0
            if names[key]._count_type is list:
                st.assert_len(ips[key], names[key].max_len(size), 0)
    # Now bounds is complete, and full-length

    # Eventually, move these recasts to top of function, see how input
    # processing logic can benefit.
    bounds = st.StatDict(bounds)
    target_counts = st.ParamDict(target_counts)
    if ips is not None:
        ips = st.ParamDict(ips)
        ## Look for regression which makes this necessary?:
        # if not st.setinbounds(ips, bounds):
        #     print("Initial parameters must lie within bounds.")
        #     return

    # mask = {}
    # for key in bounds:
    #     if isinstance(bounds[key], list):
    #         mask[key] = []
    #         for i in range(len(bounds[key])):
    #             if bounds[key][i][0] == bounds[key][i][1]:
    #                 mask[key][i] = False
    #             else:
    #                 mask[key][i] = True
    #     else:
    #         if bounds[key][0] == bounds[key][1]:
    #             mask[key] = False
    #         else:
    #             mask[key] = True

    triedparams = {}
    nametup = bounds.nametuple()

    def endalgorithm(triedparams):
        if filename is not None:
            print("Writing to file {}".format(filename))
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(("Objective_f",) + nametup + ("Observed Counts:",) + nametup)
                for key in triedparams:
                    obs_counts = triedparams[key][1]
                    if not obs_counts:  # This whole progression of logic could be improved
                        counttuple = tuple(None for _ in range(len(key)))
                    else:
                        counttuple = obs_counts.valuetuple()
                    writer.writerow((triedparams[key][0],) + key + ('',) + counttuple)

    def compute_distance(parameter_list, triedparams):
        """ Update triedparams with sampling results from parameters in
        parameter_list."""
        # print(parameter_list)
        checkthese = []  # Parameters to compute distance for
        for params in parameter_list:
            if params.valuetuple() not in triedparams:
                checkthese.append(params)
        model = Cmodel(size, checkthese)
        # model.update_prefs(output="Progress")  # For debug
        model.sample_converged(sample_period,
                               convergence_cutoff=convergence_cutoff,
                               convergence_p=convergence_p)
        for i in range(len(model.chains)):
            chain = model.chains[i]
            pars = checkthese[i]
            obs_counts = chain.avg_count_dict() # If chain didn't converge,
            # then chain has empty sample_list, and avg_count_dict() returns
            # False.
            triedparams[pars.valuetuple()] = [
                st.count_distance(target_counts, obs_counts),
                obs_counts]

    def pick_best(iplist, triedparams):
        # See which parameters from iplist work best
        compute_distance(iplist, triedparams)
        distlist = [triedparams[ips.valuetuple()][0] for ips in iplist]
        minD = min(distlist)
        index = distlist.index(minD)
        bestcounts = triedparams[iplist[index].valuetuple()][1]
        print("Best average counts from parameter set {}".format(index),
              "distance {} from desired counts: ".format(minD), bestcounts)
        print("Adjusted parameters: {}".format(iplist[index]))
        return(iplist[index], minD)

    def searchbounds(bounds, triedparams):
        # Build list of parameter sets
        iplist = [st.rand_ips(bounds) for _ in range(random_samples)]
        return(pick_best(iplist, triedparams))

    def searchstep(ips, a, triedparams):
        print("Looking for best distance minimizing search direction...")
        # Build list of parameter sets
        iplist = [deepcopy(ips)]
        for i in range(ips._inside_len()):
            for sa in [-a, a]:
                value = ips[i] + sa
                if not st.inbounds(value, bounds[i]):
                    continue
                pars = deepcopy(ips)
                pars[i] = value
                iplist.append(pars)
        # iptuplist now has unchanged parameters at index 0 only.
        return(pick_best(iplist, triedparams))

    def travelstep(ips, delta, triedparams):
        # Build list of parameter sets
        print("Checking steps in that direction ...")
        iplist = []
        indexchanged = [abs(delta[i]) > 0.0000001
                        for i in range(delta._inside_len())].index(True)
        for i in range(num_cores + 1):
            pars = ips + i * delta
            if st.inbounds(pars[indexchanged], bounds[indexchanged]):
                iplist.append(pars)
        if len(iplist) <= 1:
            print("Bounds don't allow further movement in this direction")
        return(pick_best(iplist, triedparams))

    if ips is None:
        print("Performing initial search of parameter space...")
        ips, minD = searchbounds(bounds, triedparams)
    else:
        minD = np.inf
    print("initial parameters are:", ips)
    while True:
        if minD < min_dist:
                print("Minimum distance has been achieved.")
                break
        lastips = ips.deepcopy()
        # Search step
        ips, minD = searchstep(ips, a, triedparams)
        # See if searchstep resulted in changed pars
        if ips == lastips:
            a = a / 4
            if abs(a) <= resolution:
                print("Minimum resolution has been reached.")
                break
            print("\n\n\n\nOriginal parameters seem best.",
                  "Decreasing gain value to try again.")
            continue
        delta = ips - lastips
        while True:
            lastips = ips.deepcopy()
            ips, minD = travelstep(ips, delta, triedparams)
            if (ips - lastips) != 4 * delta:
                break
    endalgorithm(triedparams)


class Model(object):
    """Model object contains sample chains, sampling preferences, and
    associated methods.

    Attributes
    ----------
    chains : array
        chain objects, one for each initial graph passed to
        model __init__ function.

    lock : None, or multiprocessing.Lock
        By default ``None``, but used by concurrent sample method to store an
        input/output lock object for use by sampling chains.

    sample_preferences : dict
        A dictionary storing 'burnin', 'sample_interval', and 'output'
        preferences.

        * 'burnin' -- (default ``n``)
            Where ``n`` is the number of nodes in sampled graphs. Burnin
            is the number of discarded iterations before sampling
            begins. Used by methods of the Chain class, but universal to all
            chains in Model.
        * 'sample_interval' -- (default ``1``)
            Number of discarded iterations between
            samples. Used by methods of the Chain class, but universal to all
            chains in Model.
        * 'output' -- (default ``None``)
            String used to determine how sampling progress
            is recorded and indicated. ``"None"`` prints basic progress,
            ``"Progress"`` prints sampling progress as it is made, and results
            in garbled output when using concurrent sampling methods,
            "Terminal" prints detailed output after sampling is finished, and
            "File" saves detailed output from each chain to a separate file
            in ./logs.
    """

    def __init__(self, initial_graphs, stats):
        """Initialize a model with initial graphs and statistic parameters.

        Parameters
        ----------
        initial_graphs : array
            Array of graph objects
            Graphs must all be of the same size. Use generator functions
            in cliquergm.graph to generate graphs.

        stats : dict
            Dictionary of statistic parameters keyed by statistic subclass
            names. If initial graphs have statistics set, then None can be
            passed to avoid overwriting those.

        Notes
        -----
        If initial graphs already have statistics associated with them, one may
        use None as the stats argument to avoid changing them.
        """
        # Save initial random state for model scope
        self.randomstate = np.random.get_state()
        random.seed(np.random.randint(1000))

        # Build chains list
        self.lock = None
        self.chains = [chain.Chain(initial_graph, stats, model=self)
                       for initial_graph in initial_graphs]

        # Set initial RNG states for each chain
        # (This only matters for multiprocessing):
        seeds = np.random.randint(1, 100000, len(self.chains))
        for i in range(len(self.chains)):
            np.random.seed(seeds[i])
            self.chains[i].randomstate = np.random.get_state()
        np.random.set_state(self.randomstate)  # reset np random seed for model scope
        # Done setting initial RNG states for each chain

        n = initial_graphs[0].number_of_nodes
        self.sample_preferences = {'burnin': initial_graphs[0].number_of_nodes,
                                   'sample_interval': int(n*(n-1)/2),
                                   'output': "None"}

    def update_prefs(self, **kwargs):
        """Update model sampling preference statistics.

        Updates values in sample_preferences dictionary.

        Keyword Arguments
        -----------------
        burnin : int
            (default ``n``)
            Where ``n`` is the number of nodes in sampled graphs.
            Iterations to discard before sampling begins.

        sample_interval : int
            (default ``1``)
            Iterations to discard between recorded samples

        output : string
            (default ``"None"``)
            Controls how sampling progress is recorded/displayed. Choose from
            "None", "Progress", "Terminal", and "File".

        Notes
        -----
        'output' preference determines how sampling progress is recorded and
        indicated. "None" prints only burnin/sampling status, "Progress" prints
        sampling progress as it is made, and results in garbled output when
        using concurrent sampling methods, "Terminal" prints detailed output
        after sampling is finished, and "File" saves detailed output from each
        chain to a separate file in ./logs.
        """
        for key in kwargs.keys():
            if key in self.sample_preferences.keys():
                self.sample_preferences[key] = kwargs[key]
            else:
                print("'{}'".format(key),
                      " is not a sample preference. Try 'burnin',",
                      "'sample_interval',", "or 'output'.")

    def sample(self, samples, concurrent=True):
        """Sample from each initial graph or chain current graph concurrently.

        Assigns a process to each chain in order to sample them concurrently.
        Array of sampled graphs is stored in each chain's sample_list

        Parameters
        ----------
        samples : int
            Number of samples to be collected.

        concurrent : boolean
            (Default ``True``)
            Whether chains should be sampled concurrently using multiprocessing.
        """
        if concurrent:
            with mp.Manager() as m, mp.Pool() as pool:
                self.lock = m.Lock()
                f = partial(_sample, samples=samples)
                # self.chains must be replaced with chain objects sampled in each
                # process. The ones in the main process's memory remain unchanged.
                # self.chains order preserved. We use _replacewith method to keep
                # object id the same, which is important for logic in CModel
                # sample_converged method.
                newchains = pool.map(f, self.chains)
                for i in range(len(self.chains)):
                    self.chains[i]._replacewith(newchains[i])
                self.lock = None
        else:
            self._csample(samples)

    def _csample(self, samples):
        """Consecutively sample from each initial graph or chain current graph.

        Array of sampled graphs is stored in each chain's sample_list

        Parameters
        ----------
        samples : int
            Number of samples to be collected.
        """
        for chain in self.chains:
            chain.sample(samples)

    def fit(self, target_counts, a):
        """Fit statistic parameters to achieve target subgraph distributions.

        This fitting algorithm doesn't work. Use the :func:`Fit() <cliquergm.fit()>` function instead.

        Currently, this method calls
        :meth:`Chain.fit() <cliquergm.chain.Chain.fit()>`
        method on each chain, so the model parameters of each chain's
        current_graph end up being different.

        Parameters
        ----------
        target_counts : dict
            Dictionary of target counts, keyed by names of subclasses of the
            :class:`~cliquergm.statistic.Statistic` class. The keys in this
            dictionary must match exactly the keys in the
            ``self.current_graph.stats`` dictionary.
            For subclasses which require a vector of counts, the vector passed
            should be exactly the right length. That is, the clique count
            vector must match the number of nodes in the graphs being sampled.

        a : float
            Initial gain value for the algorithm. Choose ``a`` to have a value
            of about ``0.1`` if the initial parameters are expected to be far
            from the fitted parameter values, and ``0.01`` if the initial
            parameter values are expected to be close to the fitted values.
        """
        prefs = self.sample_preferences
        self.update_prefs(output="None", sample_interval=1, burnin=0)
        for i in range(len(self.chains)):
            print("Fitting for chain {} ...".format(i + 1))
            self.chains[i].fit(target_counts, a)
            print("\n")
        self.update_prefs(**prefs)

    def counts(self):
        """Summarize statistic counts of stored samples in all chains.

        Returns
        -------
        count_array : numpy array
            Parameter counts are organized as ``count_array[j,t,i]`` where
            ``j`` is chain number, ``t`` is time, and ``i`` is the parameter.
        """
        return(np.array([chain.counts() for chain in self.chains]))

    def nonzero_counts(self):
        """Summarize 'significant' statistic counts of stored samples.

        Method discards parameters whose counts are close to zero, as
        determined by chain._is_zero function via chain.identify_zero_counts
        function.

        Returns
        -------
        count_array : numpy array
            Parameter counts are organized as count_array[j,t,i] where j is
            chain number, t is time, and i is the parameter.
        """
        count_array = self.counts()
        delete_array = np.concatenate(tuple(chain.identify_zero_counts(chain_counts)
                                            for chain_counts in count_array))
        count_array = np.delete(count_array, delete_array, 2)
        return(count_array)

    # ######## Convergence Diagnostic Calculation Functions ########
    def assess_mpsrf(self, start=None, step=100, plot=False, save=None,
                     figures=None):
        """Assess mpsrf in the last ``period`` samples with sliding window of
        size ``step``. This does not seem to work well, so consider using
        :meth:`cliquergm.sample_tools.paired_chains_test()` to test chain
        convergence instead."""
        if start is None:
            start = -int(len(self.chains[0].sample_list) / 2)
        count_array = self.nonzero_counts()[:, start:, :]
        timesteps = count_array.shape[1]
        results = []
        if step > timesteps:
            print("Provide a step value that is less than the number of",
                  "sampled graphs after start index.")
            return
        x_arr = np.array(range(timesteps - step))
        for x in x_arr:
            arr = list(mpsrf(count_array[:, x:x + step, :]))
            results.append(arr)
        results = np.array(results)

        if (figures is not None) or plot:
            @plot_imports
            def plt(**kwargs):
                pl, ps, *_ = kwargs['modules']
                xx = [self.chains[0].sample_list[x].iteration for x in x_arr]
                titles = ("Rp", "det(W)", "det(B_over_n)")
                fig = ps.threerowfig()
                for i in range(3):
                    fig.append_trace(pl.graph_objs.Scatter(x=xx, y=results[:, i],
                                                           name=titles[i]), i + 1, 1)
                title = ("Graph Iteration (At beginning of sliding "
                         + "window with step of "
                         + str(step * self.sample_preferences['sample_interval'])
                         + " iterations)")
                fig.layout.xaxis1.update(title=title)
                fig.layout.update(title="Multivariate PSRF Diagnostics")
                if plot:
                    pl.offline.plot(fig, filename="mpsrf.html", image=save,
                                    image_filename="mpsrf")
                if figures is not None:
                    figures.append(fig)
            plt()

        flag = True
        for i in range(1, 3):
            slc = results[:, i]
            mean = slc.mean()
            _, p1 = mannwhitneyu(
                slc, [mean for _ in range(timesteps - step)])
            _, p2 = wilcoxon(slc, [mean for _ in range(timesteps - step)])
            sumsquares = sum(abs(slc - mean)**2)/(mean * (timesteps - step))
            covar = slc.std() / mean
            print("stats for diagnostic", i)
            print("sumsquares: ", sumsquares)
            print("covar: ", covar)
            print('mwu:', p1, 'wx:', p2)
            if p1 > 0.5 or p2 > 0.5:
                flag = False
                break
        print('meanrp: ', results[:, 0].mean())
        if results[:, 0].mean() <= 1.1 and flag:
            return(True)
        else:
            return(False)

    def sample_converged(self, samples, plot=False, figures=None):
        """ Sample model chains until converged, according to the MPSRF test.

        This does not work as expected. Instead use the ``sample_converged``
        method of the ``CModel`` class.

        Returns ``samples`` graphs chosen randomly from the last 500 graphs
        sampled.
        """

        max_attempts = 10
        converged = False
        counter = 1
        while not converged:
            if counter > max_attempts:
                print("Chain failed to converge in", max_attempts,
                      "attempts, last sample iteration",
                      self.chains[0].current_graph.iteration)
                return
            print("\n Sampling stage: ", counter)
            counter += 1
            self.sample(1000)
            converged = self.assess_mpsrf(start=-500, plot=plot,
                                          figures=figures)

        print("Stop sampling, mpsrf indicates convergence.")
        print("It took {} iterations to achieve convergence".format(
              self.chains[0].sample_list[-500].iteration))

        # pick out <samples> graphs randomly from last 500:
        for ch in self.chains:
            ch.pick_samples(samples, start=-500)

    def mpsrf(self):
        """Calculate multivariate psrf for sampled graphs stored in chains.
        See model.mpsrf for information.
        """
        return(mpsrf(self.nonzero_counts()))


class Cmodel(Model):
    def __init__(self, size, statslist):
        """Initialize a model with two chains for each parameter set supplied:
        one sampling from a complete graph and one from an empty graph.

        Parameters
        ----------
        size : int
            Size of graphs to be sampled

        statslist : list
            List of dictionaries of statistic parameters keyed by
            statistic subclass names.
        """
        # Save initial random state for model scope
        self.randomstate = np.random.get_state()
        random.seed(np.random.randint(1000))

        # Build chains list
        self.lock = None
        graphs = [gr.empty_graph(size), gr.complete_graph(size)]
        self.chains = []
        for stats in statslist:
            for g in graphs:
                self.chains.append(chain.Chain(g, stats, self))
        # Set initial RNG states for each chain
        # (This only matters for multiprocessing):
        seeds = np.random.randint(1, 100000, len(self.chains))
        for i in range(len(self.chains)):
            np.random.seed(seeds[i])
            self.chains[i].randomstate = np.random.get_state()
        np.random.set_state(self.randomstate)  # reset np random seed for model
        # scope. Now done setting initial RNG states for each chain

        self.sample_preferences = {'burnin': 0,
                                   'sample_interval': int(size * (size-1) / 2),
                                   'output': "None"}
        self.size = size
        self.converged = False

    def sample_converged(self, sample_size, convergence_cutoff=6,
                         concurrent=True, convergence_p=0.05):
        """ Sample until chains which share parameters and start from initial
        empty and complete graphs appear well-mixed, according to the Paired
        Chains test (TODO: Cite).

        After convergence has been achieved for each chain pair,
        only one chain for each parameter set is retained, and the last
        sample_size samples from both chains are mixed and stored in the
        retained chain's sample_list attribute. This method may be run only
        once. If more samples are needed, use `sample` method. If chains for a
        parameter set never converge, then they will be left with an empty
        sample_list.

        Parameters
        ----------
        sample_size : int
            The length of sampling which should be completed between tests for
            convergence.

        convergence_cutoff : int
            (default ``6``)
            The method will give up attempting to sample a pair of chains to
            convergence if the Paired Chains test doesn't detect convergence
            after `convergence_cutoff * sample_size` samples.
        """
        if self.converged:
            raise UserWarning("Sample_converged method was already called on\
            this Cmodel instance.")
        else:
            self.converged = True
        num_chains = int(len(self.chains)/2)
        num_converged = 0
        print("Sampling {} parameter sets".format(num_chains),
              "\nConverged:", end=' ')
        self.sample(self.size)  # This should be enough of a burnin, since
        # samples are subject to sample_interval.
        frozenchains = self.chains  # to preserve original chain order
        for _ in range(convergence_cutoff):
            self.sample(sample_size, concurrent=concurrent)
            for chain in self.chains:
                chain.prune_samples(sample_size)
            tempchains = []  # Pick out those chains which need more sampling
            for i in range(int(len(self.chains)/2)):  # Prone to frameshift err
                if st.paired_chains_test(self.chains[2*i].sample_list,
                                         self.chains[2*i+1].sample_list,
                                         p=convergence_p):
                    tempchains.extend([self.chains[2*i], self.chains[2*i+1]])
            self.chains = tempchains  # These will be sampled in next loop iter
            # Some printing:
            temp_num_converged = num_chains - int(len(self.chains)/2)
            for i in range(num_converged + 1, temp_num_converged + 1):
                print(i, end=' ')
            num_converged = temp_num_converged

            if len(self.chains) == 0:
                break
        print('')
        for chain in self.chains:
            # For the chains that never converged
            chain.sample_list = []
        self.chains = []
        # self.chains should be an empty list
        for i in range(int(len(frozenchains)/2)):
            # print(st.avg_count_dict(frozenchains[2*i].sample_list),
            # '\n', st.avg_count_dict(frozenchains[2*i+1].sample_list),'\n')
            frozenchains[2*i].sample_list.extend(frozenchains[2*i+1].sample_list)
            random.shuffle(frozenchains[2*i].sample_list)
            self.chains.append(frozenchains[2*i])


# ######## Convergence Diagnostic Calculation Functions ########
def mpsrf(sequence_counts):
    r""" Calculate multivariate potential reduction factor.

    Uses Gelman and Rubin multivariate method [1]_ to assess convergence
    between sample sets from different initial conditions.
    Returns a multivariate potential scale reduction diagnostic
    Rp, a matrix V and a matrix W. Ignores statistics whose counts are too
    frequently zero.

    Parameters
    ----------
    sequence_counts : numpy array
        Parameter counts organized such that sequence_counts[j,t,i] gives
        i\ :sup:`th` parameter at time t in chain j.

    Returns
    -------
    Rp : float
        multivariate psrf diagnostic for convergence

    det(W) : float
        Determinant of the within-sequence statistic count covariance matrix.

    det(B_over_n) : float
        Determinant of the between-sequence statistic count covariance matrix.

    Notes
    -----
    Convergence is implied when determinants of W and B_over_n stabilize,
    and when Rp is near 1.

    References
    ----------
    ..  [1] Brooks, S. P., & Gelman, A. (1998).
        General methods for monitoring convergence of iterative simulations.
        *Journal of computational and graphical statistics*, 7(4), 434-455.
        <https://doi.org/10.1080/10618600.1998.10474787>
    """
    m, n, p = sequence_counts.shape[0:3]

    W = np.zeros((p, p))
    B_over_n = np.zeros((p, p))

    for j in range(m):
        # add to B_over_n
        v = (np.mean(sequence_counts[j, :, :], axis=0)
             - np.mean(np.mean(sequence_counts, axis=0), axis=0))
        B_over_n += np.outer(v, v)

        for t in range(n):
            # add to W
            v = sequence_counts[j, t, :] - np.mean(
                sequence_counts[j, :, :], axis=0)
            W += np.outer(v, v)
    # Scale B_over_n and W
    # print(np.linalg.det(B_over_n))
    W = W / (m * (n - 1))
    B_over_n = B_over_n / (m - 1)

    Rp = ((n - 1) / n + ((m + 1) / m) *
          np.max(np.linalg.eigvalsh(np.matmul(np.linalg.inv(W), B_over_n))))
    # print(B_over_n)

    return(Rp, np.linalg.det(W), np.linalg.det(B_over_n))
