import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Sequence
import warnings
def sub_topoplot(ax, array, x_i=0, line_i=1, av_i=None, mx=1, my=1, dx=0, dy=0, scale=0.2, **kwargs):
    ax_i = np.array([x_i, line_i, av_i])
    if array.ndim <= ax_i[pd.notnull(ax_i)].max():
        print('Error! There is no axis %d in "array".'%ax_i[pd.notnull(ax_i)].max())
        return
    nonuse_i = list(set(range(array.ndim)).difference(set(ax_i[pd.notnull(ax_i)])))
    if not all([array.shape[i] ==1 for i in nonuse_i]):
        print('Error! Size of axis %s must be 1.'%nonuse_i)
        return
    array1 = np.transpose(array, ax_i[pd.notnull(ax_i)].tolist()+nonuse_i) * my * scale
    x = np.arange(array1.shape[0])*mx+dx
    lines = [0] * array.shape[line_i]
    if kwargs.get('color') is None:
        colors = [plt.get_cmap('tab10')(i%10) for i in range(len(lines))]
    else:
        colors = kwargs.get('color')
    for i in range(len(lines)):
        if array1.ndim > 2 and array1.shape[2] > 1:
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y = np.nanmean(array1[:,i,], axis=1)
            y_sd = np.nanstd(array1[:,i,], axis=1)
            ax.fill_between(
                x, y+dy-y_sd, y+dy+y_sd
                ,facecolor=colors[i], alpha=0.1)
        else:
            y = array1[:,i]
        lines[i] = ax.plot(
            x
            ,y+dy
            ,linewidth=0.3
            ,c=colors[i]
        )[0]
    xlim = (dx, array1.shape[0]*mx+dx)
    ylim = (dy-my*scale/float(2), dy+my*scale/float(2))
    if kwargs.get('hlines') is not None:
        for hline in np.atleast_1d(kwargs.get('hlines')):
            [plt.arrow(xlim[0], y*my+dy, xlim[1] - xlim[0], 0, lw=0.1, color='gray', ls='dashed', head_width=0, head_length=0) for y in np.atleast_1d(hline)]
    if kwargs.get('vlines') is not None:
        for vline in np.atleast_1d(kwargs.get('vlines')):
            [plt.arrow(x*mx+dx, ylim[0], 0, ylim[1]-ylim[0], lw=0.1, color='gray', ls='dashed', head_width=0, head_length=0) for x in np.atleast_1d(vline)]
    #ax.fill_between(
    #                x = x, y1=ylim[0], y2=ylim[1]
    #                ,where=[False]*100+[True]*kwargs.get('stim_duration')+[False]*(array1.shape[0]-100-kwargs.get('stim_duration'))
    #                ,facecolor='gray', alpha=0.2)
    head_size = np.nanmean([xlim[1]-xlim[0], ylim[1]-ylim[0]])/30
    plt.arrow(xlim[0], ylim[0], xlim[1]-xlim[0], 0, lw=0.1, color="k", head_width=head_size, head_length=head_size)
    plt.arrow(xlim[0], ylim[0], 0, ylim[1]-ylim[0], lw=0.1, color="k", head_width=head_size, head_length=head_size)
    if kwargs.get('text'):
        plt.text(xlim[0]+(xlim[1]-xlim[0])/2, ylim[1],  kwargs.get('text'), fontsize=4, horizontalalignment='center')
    return lines

def topoplot(list_of_array, layout='grid', mx=None, my=None, subtopofunc=None, **kwargs):
    if isinstance(layout, str) and layout == 'grid':
        grid_size = int(np.ceil(np.sqrt(len(list_of_array))))
        layout = np.stack((
            np.tile(np.arange(grid_size),grid_size)
            ,np.tile(np.arange(grid_size), (grid_size,1)).T.flatten()
        )).T[:len(list_of_array)+1,:]
    if mx is None:
        x_i_ = kwargs.get('x_i') if kwargs.get('x_i') is not None else 0
        x_max_ = np.nanmax([array.shape[x_i_] for array in list_of_array])
        mx = np.ptp(layout[:,0])/np.sqrt(len(layout)) / x_max_
    if my is None:
        line_max_ = np.nanmax([np.ptp(array) for array in list_of_array])
        my = np.ptp(layout[:,1])/np.sqrt(len(layout)) / line_max_
    if subtopofunc is None:
        subtopofunc = sub_topoplot
    if kwargs.get('texts') is None:
        texts = [None] * len(list_of_array)
    else:
        texts = kwargs.get('texts')
    fig = plt.figure(dpi=320)
    ax = plt.subplot(111)
    lineses = [0] * len(list_of_array)
    for i, array in enumerate(list_of_array):
        lineses[i] = subtopofunc(ax, array, mx=mx, my=my, dx=layout[i, 0], dy=layout[i, 1], text=texts[i], **kwargs)
    if kwargs.get('title'):
        plt.title(kwargs.get('title'))
    try:
        if kwargs.get('labels') is not None:
            plt.legend(handles=lineses[0], labels=kwargs.get('labels'), fontsize='x-small')
    except:
        pass
    return ax, lineses
