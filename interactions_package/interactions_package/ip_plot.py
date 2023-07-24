##
## Plotting functions for use across notebooks
## In particular, custom (line) 2-D ALE plots,
## bar plots, and Shapley/ALE comparison plots
##

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import shap
from PyALE import ale

def plot_defaults():
    """ Set default plot parameters"""
    plt.style.use('seaborn-v0_8-white')
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams.update({'axes.titlesize': 16})

def plot_basic_bar(data, y, 
                   label = None,
                   n_bars = 10,
                   figsize = None,
                   ylabel = None,
                   title=None,
                   do_sort = False):
    """ Create a basic bar plot for a Pandas dataframe."""
    
    if do_sort:
        data = data.copy().sort_values(y, ascending=False)
        
    if label != None:
        data = data.copy().set_index(label)
        
    if ylabel == None:
        ylabel = data[y].name
        
    # Set figsize if not explicit
    if figsize == None:
        figsize = (4, n_bars/3.3)
        
    fig, ax = plt.subplots()
    
    data.head(n_bars)[[y]] \
        .plot(kind='barh', legend=None, figsize=figsize, ax=ax)
    
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_ylabel(None)
    ax.set_xlabel(ylabel)
    
    return fig

def plot_default_scale(x_data, thresh=3):
    """Simple function which decides whether to plot
    data using a linear or log scale.  If the skewness
    is past a threshold, use a log scale"""
    if (scipy.stats.skew(x_data) > thresh):
        return 'log'
    return 'linear'


#
# SHAP and ALE comparison plot
#

def plot_comp_ale_shap(ale_data, shap_data, color_categories = [36, 60],
                       features = ['int_rate', 'term'], title = None):
    """ Prints a plot comparing SHAP and ALE data"""
    
    cmap = mpl.cm.coolwarm
    cnorm  = mpl.colors.Normalize(vmin=0, vmax= len(color_categories) -1)
    color_scalar_map = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    color_dict = {color_categories[i]: color_scalar_map.to_rgba(i) 
                  for i in range(0, len(color_categories))}
    
    fig, ax = plt.subplots(1, 2, figsize = (12,3), sharex = True)
    
    ale_data[color_categories].plot(legend=None, ax=ax[0], cmap=cmap)
    ax[0].set_title('ALE')
    ax[0].set_ylabel(None)

    for c in color_categories:
        shap_data[shap_data[features[1]] == c][[features[0], 'shap']] \
            .plot(x=features[0], y='shap', kind='scatter', ax=ax[1], color=color_dict[c],
                 label=c)
    ax[1].set_title('SHAP')  
    ax[1].set_ylabel(None)
    ax[1].legend(bbox_to_anchor=(1.2, 1.05))
    
    if title is not None:
        fig.suptitle(title)
    
    return fig

#
# SHAP interaction plotting
#

def plot_shap_term_int(data, feature, outfile=None):
    
    # Get color map for term
    cmap = mpl.cm.coolwarm
    color_categories = data['term'].value_counts().index
    cnorm  = mpl.colors.Normalize(vmin=0, vmax= len(color_categories) - 1)
    color_scalar_map = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    color_dict = {color_categories[i]: color_scalar_map.to_rgba(i) 
                  for i in range(0, len(color_categories))}
    
    plot_feature = 'shap_' + feature
    
    fig, ax = plt.subplots()

    for key, group in data.groupby('term'):
        group.plot(x='int_rate', y=plot_feature, kind='scatter', ax=ax,
              color=color_dict[key], label=key)
        ax.legend(frameon=True, title='term')
        ax.set_ylabel('overall SHAP value for ' + feature)
    
    if outfile is not None:
        fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    
    return fig

def shap_dependence_plot(shap_data, X, features,
                             y_label= 'SHAP value',
                             title = None,
                             title_prefix = None,
                             output_path = None,
                            figsize = (7,4)):
    """ Create a SHAP dependence plot. Use the default SHAP
    package plot, but automatically change the X axis to log 
    if needed. Also modifies the title, and optionally saves the plot
      Inputs:
        shap_data:  Shapley interaction data - numpy array
        X: Feature values associated with shap_data, e.g. the training data
            including only predictor features.  Pandas dataframe
        features:  Tuple containing the interaction (or main effect) to plot
        title:  Plot title.  Set to "features" to use the feature parameter
        title_prefix: Prefix string to print before the title.  
                  None by default
        output_path:  If not None, the figure is saved to this location
      Value:  Figure containing the modified dependence plot
    """
    
    # Get standard title, if requested
    if title == 'features':
        if (len(features) >= 2) & (features[0] != features[1]):
            title = ':'.join(list(features))
        else:
            title = features[0]
    if title_prefix is not None:
        title = title_prefix + title
        
    # If needed, reverse the feature order for interaction plots
    if (len(features) == 2) & (features[0] != features[1]):
        levels_1 = len(X[features[0]].sample(100).value_counts())
        if levels_1 < 4:
            features = features[::-1]
    
    # Get but do not show default plot
    fig, ax = plt.subplots(figsize=figsize)
    shap.dependence_plot(features, shap_data, X, ax=ax, show=False)
    
   
    # Modify x axis scale if necessary
    x_feature = ax.get_xlabel()
    ax.set_xscale(setup.plot_default_scale(X[x_feature]))
    
    # Modify y axis scale if necessary
    y_vals = [x[1] for x in ax.collections[0].get_offsets()]
    y_scale = setup.plot_default_scale(y_vals)
    if (y_scale == 'log'):
        y_label = y_label + ' (log scale)'
        ax.set_yscale(y_scale)
    
    ax.autoscale(axis='both')
    
    # Change title and axis label
    plt.title(title)
    ax.set_ylabel(y_label)
    
    # Optionally save the plot
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
    
    return fig


#
# ALE Plotting
#


def ale_2way_plot(data, 
                  cmap = mpl.cm.coolwarm,
                  figsize=(7,4),
                  y_label= '2-way ALE',
                  title = None,
                  title_prefix = None,
                  legend_colobar_thresh = 4):
    """ Creates a 2D ALE line plot, where columns are shown as
    separate lines with different colors. Log scaling is applied
    to the x axis, if ALE buckets seem to have large gaps.
      Inputs:
        data:     2 way ALE data from PyALE.ale
        cmap:     color map for multiple lines
        figsize:  figsize for pandas.DataFrame.plot
        y_label:  y axis text label
        title:    plot title.  Set to "features" to 
                  use a string like "feature1:feature2"
        title_prefix: Prefix string to print before the title.  
                  None by default.
        legend_colobar_thresh:  If we have more than this number
                  of lines to plot, use a colorbar legend.  
                  Otherwise,use a regular legend.
      Value:
          Figure containing the 2 way plot
    """
    
    # Get standard title, if requested
    if title == 'features':
        title=str(data.index.name) + ':' + str(data.columns.name)
    
    if title_prefix is not None:
        title = title_prefix + title
    
    # Orient the data so columns have the feature with fewer buckets
    if len(data) < len(data.columns):
        data = data.copy().transpose()
    
    # Colors array for plot
    n = len(data.columns)
    colors = cmap(np.linspace(0,1,n))

    fig, ax = plt.subplots()
    
    data.plot(color=colors, legend=None, figsize=figsize, ax=ax)
    ax.set_ylabel(y_label)
    plt.title(title)
    bounds = [float(c) for c in data.columns]
    
    # Add colorbar legend, if we have enough columns
    if len(data.columns) >= legend_colobar_thresh:
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                       ax=ax, orientation='vertical')
        clb.set_label(data.columns.name, labelpad=20, rotation=-90, 
                     horizontalalignment='right', y=0.4)
    else:
        ax.legend(title=data.columns.name)
        
    # X-axis log scaling, if point spacing is irregular
    plt.xscale(plot_default_scale(list(data.index)))
    
    return fig