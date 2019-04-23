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

"""Templates for Plotly graph creation.

All templates are wrapped in functions to avoid needing to use cp.deepcopy
elsewhere.
"""

from functools import wraps
import math
import numpy as np


def plot_imports(func):
    """Decorator to attempt import of Plotly Module for plotting.
    Imports
    -------
    pl : plotly
    ps : cliquergm.plotstyles

    Notes
    -----
    Imports are passed to wrapped function in kwargs['modules'] as tuple
    (pl, ps). Wrapped function must accommodate this keyword argument,
    either with '**kwargs' or with 'modules' keyword argument.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        imgsize = [1600, 1200]
        # In the future, this should take all arguments specific to plotly
        # plot function, like image_height, image_width, etc. in kwargs.
        try:
            import plotly as pl
            import plotly.figure_factory as ff
            # import plotly.offline as ply
            # import plotly.graph_objs as plygo
            # import plotly.io as pio
            import cliquergm.plotstyles as ps
        except ImportError:
            print("Plotting by function or method '{}'".format(func.__name__),
                  " requires module 'plotly'.")
            return
        if 'imgsize' not in kwargs:
            kwargs.update({"imgsize": imgsize})
        kwargs.update(modules=(pl, ps, ff))
        return(func(*args, **kwargs))
    return(wrapper)


try:
    import plotly.graph_objs as go
    import plotly.tools as pt

    # ####### Graph Plotting ########
    # Layout
    def graphaxis():
        return({'showline': False,
                'zeroline': False,
                'showgrid': False,
                'showticklabels': False,
                'title': '',
                'showspikes': False})

    def graphlayout():
        return(go.Layout(showlegend=False,
                         xaxis=graphaxis(),
                         yaxis=graphaxis(),
                         hovermode='closest'))

    def graphlayout3d():
        return(go.Layout(showlegend=False,
                         scene=go.Scene(xaxis=go.YAxis(graphaxis()),
                                        yaxis=go.XAxis(graphaxis()),
                                        zaxis=go.ZAxis(graphaxis())),
                         hovermode='closest'))

        # Data
    def graphedge():
        return({'width': 0.5, 'color': '#888'})

    def nodemarker():
        return({'color': '#c6272f',
                'size': 10,
                'line': dict(width=2)})

    def edgescatter(x=[], y=[]):
        return(go.Scatter({'x': x, 'y': y,
                           'line': graphedge(),
                           'hoverinfo': 'none',
                           'mode': 'lines'}))

    def nodescatter(x=[], y=[], text=[]):
        return(go.Scatter({'x': x, 'y': y,
                           'text': text,
                           'mode': 'markers',
                           'hoverinfo': 'text',
                           'marker': nodemarker()}))

    def edgescatter3d(x=[], y=[], z=[]):
        return(go.Scatter3d({'x': x, 'y': y, 'z': z,
                             'line': graphedge(),
                             'hoverinfo': 'none',
                             'mode': 'lines'}))

    def nodescatter3d(x=[], y=[], z=[], text=[]):
        return(go.Scatter3d({'x': x, 'y': y, 'z': z,
                             'text': text,
                             'mode': 'markers',
                             'hoverinfo': 'text',
                             'marker': nodemarker()}))

    # ######## Convergence Plotting ########
    def threerowfig():
        return(pt.make_subplots(rows=3,
                                cols=1,
                                shared_xaxes=True,
                                print_grid=False))

    # ######## Fit Plotting ########
    def outlined_histogram(data, name):
        numbins = math.ceil(max(data) - min(data))
        minbin = math.floor(min(data))
        binedges = np.array(list(range(minbin, minbin + numbins + 2))) - 0.5
        dist, x = np.histogram(data,
                               bins=binedges,
                               density=True,)
        realbinwidth = x[1] - x[0]
        bincenters = x + realbinwidth/2
        if dist[0] != 0:
            bincenters = np.insert(bincenters, 0, bincenters[0] - realbinwidth)
            dist = np.insert(dist, 0, 0)
        dist = np.append(dist, 0)
        trace = go.Scatter(x=bincenters,
                           y=dist,
                           mode='lines',
                           name=name,
                           line=dict(shape='hvh'))
        return(trace)

    def barplot(x=[], y=[], text=[]):
        return(go.Bar({'x': x, 'y': y}))

    def tworowfig(titles=(), shared_xaxes=False):
        return(pt.make_subplots(rows=2,
                                cols=1,
                                subplot_titles=titles,
                                shared_xaxes=shared_xaxes,
                                print_grid=False))

    def baselayout():
        return(go.Layout(barmode='overlay',
                         font=dict(size=20),
                         xaxis=dict(zeroline=False),
                         yaxis=dict(zeroline=False),
                         legend=dict(
                             x=0.8,
                             y=0.95,
                             traceorder='normal',
                             bgcolor='#E2E2E2',
                             bordercolor='#FFFFFF',
                             borderwidth=2)))

except ImportError:
    pass
