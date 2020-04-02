import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import re
from collections import Sequence

def xplot(ax, array1, av_options={'mean':True, 'std':True, 'each':False}, **kwargs):
    x = np.arange(array1.shape[0])
    #lines = [0] * array.shape[line_i]
    lines = [0] * array1.shape[1]
    colors = kwargs.get('colors')
    if colors is None:
        colors = [plt.get_cmap('tab10')(i%10) for i in range(len(lines))]
    for i in range(len(lines)):
        if array1.shape[2] > 1:
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y = np.nanmean(array1[:,i,], axis=1)
            if av_options.get('mean', True):
                lines[i] = ax.plot(x, y, linewidth=0.3, c=colors[i])[0]
            y_sd = np.nanstd(array1[:,i,], axis=1)
            if av_options.get('std', True):
                ax.fill_between(x, y-y_sd, y+y_sd
                    ,facecolor=colors[i], alpha=0.1)
            if av_options.get('each', False):
                ax.plot(
                    x
                    ,array1[:,i,:]
                    ,linewidth=0.3
                    ,c=colors[i]
                    ,alpha=0.1
                )
        else:
            lines[i] = ax.plot(
                x
                ,array1[:,i]
                ,linewidth=0.3
                ,c=colors[i]
            )[0]
    return lines

def sub_topoplot(ax, array, x_i=0, line_i=1, av_i=2
    , plot_func=xplot
    , av_options={'mean':True, 'std':True, 'each':False}, **kwargs):
    mpl.rcParams['axes.linewidth'] = 0.2
    if isinstance(array, pd.DataFrame):
        #print('Warning, array is a DataFrame.')
        array1 = array
    else:
        if array.ndim==1:
            array = np.transpose(np.atleast_3d(array), [1,0,2])
        else:
            array = np.atleast_3d(array)
        if x_i > 2:
            print('Error! x_i: %d. x_i must be 0, 1 or 2.'%x_i)
            return
        if line_i > 2:
            print('Error! line_i: %d. line_i must be 0, 1 or 2.'%line_i)
            return
        if av_i > 2:
            print('Error! av_i: %d. av_i must be 0, 1 or 2.'%av_i)
            return
        if x_i == line_i or x_i == av_i or line_i == av_i:
            print('Error! x_i: %d, line_i:%d, av_i;%d. x_i, line_i and av_i must be different.'%(x_i,line_i,av_i))
            return
        array1 = np.transpose(array, [x_i, line_i, av_i])
    lines = plot_func(ax, array1, av_options=av_options, **kwargs)
    
    if kwargs.get('texts') is not None:
        [_set(ax.text, text_) for text_ in kwargs.get('texts') if isinstance(text_,dict) and all([xys in text_.keys() for xys in ('x', 'y', 's')])]
    _set(ax.set_xticks,kwargs.get('xticks'))
    _set(ax.set_xticklabels,kwargs.get('xticklabels'))
    _set(ax.set_yticks,kwargs.get('yticks'))
    _set(ax.set_yticklabels,kwargs.get('yticklabels'))
    _set(ax.set_xlabel,kwargs.get('xlabel'))
    _set(ax.set_ylabel,kwargs.get('ylabel'))
    if kwargs.get('xlim') is not None:
        ax.set_xlim(kwargs.get('xlim'))
    if kwargs.get('ylim') is not None:
        ax.set_ylim(kwargs.get('ylim'))
    if kwargs.get('zorder') is not None:
        ax.set_zorder(kwargs.get('zorder'))
    if kwargs.get('hlines') is not None:
        for hline in np.atleast_1d(kwargs.get('hlines')):
            [ax.arrow(ax.get_xlim()[0], hline_y, np.ptp(ax.get_xlim()), 0, lw=0.1, color='gray', ls='dashed', head_width=0, head_length=0) for hline_y in np.atleast_1d(hline)]
    if kwargs.get('vlines') is not None:
        for vline in np.atleast_1d(kwargs.get('vlines')):
            [ax.arrow(vline_x, ax.get_ylim()[0], 0, np.ptp(ax.get_ylim()), lw=0.1, color='gray', ls='dashed', head_width=0, head_length=0) for vline_x in np.atleast_1d(vline)]
    #[tick.label.set_fontsize(2) for tick in ax.xaxis.get_major_ticks()]
    ax.tick_params(direction='in', length=1, width=0.2, pad=0.4)
    #[tick.label.set_fontsize(2) for tick in ax.yaxis.get_major_ticks()]
    _set(ax.set_title,kwargs.get('title'))
    return lines

def _isoverlap(s1s2):#s1_left, s1_right, s1_bottom, s1_top, s2_left, s2_right, s2_bottom, s2_top):
    s1 = s1s2[0]
    s2 = s1s2[1]
    return (\
        (s2[0] <s1[0] and s1[0] < s2[1])\
        or (s2[0] <s1[1] and s1[1] < s2[1])\
        or (s1[0] <s2[0] and s2[0] < s1[1])\
        or (s1[0] <s2[1] and s2[1] < s1[1]))\
        and\
        ((s2[2] <s1[2] and s1[2] < s2[3])\
        or (s2[2] <s1[3] and s1[3] < s2[3])\
        or (s1[2] <s2[2] and s2[2] < s1[3])\
        or (s1[2] <s2[3] and s2[3] < s1[3]))

def _find_best_swh(layout, wh_rate=1):
    from itertools import combinations
    def _s1s2(layout, sw, sh):
        return (layout[0], layout[0]+sw, layout[1], layout[1]+sh), (layout[2], layout[2]+sw, layout[3], layout[3]+sh)
    train = [0.000, 1]
    pair_layout = pd.DataFrame(list(combinations(range(len(layout)),2)), columns=[ 'ch1', 'ch2'])
    pair_layout['s1_x'] = layout[pair_layout.ch1.values,0]
    pair_layout['s1_y'] = layout[pair_layout.ch1.values,1]
    pair_layout['s2_x'] = layout[pair_layout.ch2.values,0]
    pair_layout['s2_y'] = layout[pair_layout.ch2.values,1]
    pair_layout = pair_layout[np.abs(pair_layout.s1_x - pair_layout.s2_x) < 0.3]
    pair_layout = pair_layout[np.abs(pair_layout.s1_y - pair_layout.s2_y) < 0.3]
    for n in range(7):
        isoverlaps = [_isoverlap(_s1s2(layout_, np.mean(train), np.mean(train)*wh_rate)) for layout_ in pair_layout.loc[:,['s1_x','s1_y','s2_x','s2_y']].values]
        if any(isoverlaps):
            train[1] = np.mean(train)
        else:
            train[0] = np.mean(train)
    return train[0], train[0]*wh_rate

def _get(x,i):
    if isinstance(x, Sequence):
        return x[int(i%len(x))]
    else:
        return x

def _gets(kwargs, i):
    new_keys = []
    new_vals = []
    for key, val in zip(kwargs.keys(), kwargs.values()):
        if (len(key)>3) & (key[:3]=='sup'):
            continue
        if key[-1] == 's':
            new_vals.append(_get(kwargs.get(key, None), i))
            if (len(key)>3) & (key[-3:] == 'ses'):
                new_keys.append(key[:-2])
            else:
                new_keys.append(key[:-1])
        else:
            new_keys.append(key)
            new_vals.append(val)
    return dict(zip(new_keys, new_vals))

def _set(func, x):
    if x is not None:
        if isinstance(x, dict):
            return func(**x)
        else:
            return func(x)

def topoplot(list_of_array, layout='grid', sws=None, shs=None, subtopofunc=None, **kwargs):
    if isinstance(layout, str):
        if layout == 'grid':
            grid_size_x = int(np.ceil(np.sqrt(len(list_of_array))))
            grid_size_y = grid_size_x
        elif 'x' in layout.lower() or 'y' in layout.lower():
            mx = re.search('x(\d+)', layout.lower())
            if mx:
                grid_size_x = int(mx.group(1))
            my = re.search('y(\d+)', layout.lower())
            if my:
                grid_size_y = int(my.group(1))
            if mx and not my:
                grid_size_y = int(np.ceil(len(list_of_array) / grid_size_x))
            elif not mx and my:
                grid_size_x = int(np.ceil(len(list_of_array) / grid_size_y))
        if sws is None:
            sws = 1/grid_size_x
        layout = np.stack((
            np.tile(np.arange(grid_size_x),grid_size_y)
            ,np.tile(np.arange(grid_size_y), (grid_size_x,1)).T.flatten()[::-1]
        )).T[:len(list_of_array)+1,:]
    layout = (layout - np.min(layout, axis=0)) / np.ptp(layout, axis=0)
    layout[np.isnan(layout)] = 0
    if sws is None and shs is None:
        sws, shs = _find_best_swh(layout, kwargs.get('wh', 1)*(kwargs.get('figsize', (6,4))[0]/kwargs.get('figsize', (6,4))[1]))
    elif shs is None:
        shs =  sws * kwargs.get('wh', 1)*(kwargs.get('figsize', (6,4))[0]/kwargs.get('figsize', (6,4))[1])
    elif sws is None:
        sws = shs / kwargs.get('wh', 1)*(kwargs.get('figsize', (6,4))[0]/kwargs.get('figsize', (6,4))[1])
    if subtopofunc is None:
        subtopofunc = sub_topoplot
    fig = plt.figure(dpi=kwargs.get('dpi', 320), figsize=kwargs.get('figsize', (6,4)))
    lineses = [None] * len(list_of_array)
    for i, array in enumerate(list_of_array):
        sw = _get(sws,i) * _get(kwargs.get('subplot_scales', 1), i)
        sh = _get(shs,i) * _get(kwargs.get('subplot_scales', 1), i)
        ax = fig.add_axes(list(layout[i,:]) + [sw, sh])
        lineses[i] = subtopofunc(ax, array
            ,ax_i=i
            ,**_gets(kwargs, i)
#            ,texts=_get(kwargs.get('textses', None), i)
#            ,xticks=_get(kwargs.get('xtickses', None), i)
#            ,xticklabels=_get(kwargs.get('xticklabelses', None), i)
#            ,yticks=_get(kwargs.get('ytickses', None), i)
#            ,yticklabels=_get(kwargs.get('yticklabelses', None), i)
#            ,xlim=_get(kwargs.get('xlims', None), i)
#            ,ylim=_get(kwargs.get('ylims', None), i)
#            ,colors=_get(kwargs.get('colorses', None), i)
#            ,zorder=_get(kwargs.get('zorders', None), i)
#            ,title=_get(kwargs.get('titles', None), i)
#            ,**kwargs
            )
    if pd.isnull(lineses).all():
        return None
    else:
        if kwargs.get('suptitle') is not None:
            fig.suptitle(kwargs.get('suptitle'), y = np.max(layout[:,1])+sh*1.5, verticalalignment='bottom')
        #try:
        if kwargs.get('suplabels') is not None:
            plt.legend(handles=lineses[pd.notnull(lineses).nonzero()[0][0]], labels=kwargs.get('suplabels'), loc=kwargs.get('suplegend_loc', 3), fontsize=5, bbox_to_anchor=(np.max(layout, axis=0) - layout[i,:] + np.array([sw, sh])) // np.array([sw, sh]))
        #except:
        #    pass
    #    if kwargs.get('savefig') is not None:
        _set(fig.savefig, kwargs.get('savefig'))
        if kwargs.get('show') is not None and not kwargs.get('show'):
            plt.close(fig)
        return fig
