import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import re
from collections import Sequence

def sub_topoplot(ax, array, x_i=0, line_i=1, av_i=2
    , av_options={'mean':True, 'std':True, 'each':False}, **kwargs):
    mpl.rcParams['axes.linewidth'] = 0.2
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
    '''
    ax_i = np.array([x_i, line_i, av_i])
    if array.ndim <= ax_i[pd.notnull(ax_i)].max():
        print('Error! There is no axis %d in "array".'%ax_i[pd.notnull(ax_i)].max())
        return
    nonuse_i = list(set(range(array.ndim)).difference(set(ax_i[pd.notnull(ax_i)])))
    if not all([array.shape[i] ==1 for i in nonuse_i]):
        print('Error! Size of axis %s must be 1.'%nonuse_i)
        return
    array1 = np.transpose(array, ax_i[pd.notnull(ax_i)].tolist()+nonuse_i)
    '''
    x = np.arange(array1.shape[0])
    lines = [0] * array.shape[line_i]
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
        '''
        if array1.ndim > 2 and array1.shape[2] > 1:
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y = np.nanmean(array1[:,i,], axis=1)
            y_sd = np.nanstd(array1[:,i,], axis=1)
            ax.fill_between(
                x, y-y_sd, y+y_sd
                ,facecolor=colors[i], alpha=0.1)
        else:
            y = array1[:,i]

        lines[i] = ax.plot(
            x
            ,y
            ,linewidth=0.3
            ,c=colors[i]
        )[0]
        '''
    if kwargs.get('texts') is not None:
        [_set(ax.text, text_) for text_ in kwargs.get('texts') if isinstance(text_,dict) and all([xys in text_.keys() for xys in ('x', 'y', 's')])]
    if kwargs.get('xticks') is not None:
        _set(ax.set_xticks,kwargs.get('xticks'))
    if kwargs.get('xticklabels') is not None:
        _set(ax.set_xticklabels,kwargs.get('xticklabels'))
    if kwargs.get('yticks') is not None:
        _set(ax.set_yticks,kwargs.get('yticks'))
    if kwargs.get('yticklabels') is not None:
        _set(ax.set_yticklabels,kwargs.get('yticklabels'))
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
    if kwargs.get('title') is not None:
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
        return x[i]
    else:
        return x

def _set(func, x):
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
    lineses = [0] * len(list_of_array)
    for i, array in enumerate(list_of_array):
        sw = _get(sws,i) * _get(kwargs.get('subplot_scales', 1), i)
        sh = _get(shs,i) * _get(kwargs.get('subplot_scales', 1), i)
        ax = fig.add_axes(list(layout[i,:]) + [sw, sh])
        lineses[i] = subtopofunc(ax, array
            ,ax_i=i
            ,texts=_get(kwargs.get('textses', None), i)
            ,xticks=_get(kwargs.get('xtickses', None), i)
            ,xticklabels=_get(kwargs.get('xticklabelses', None), i)
            ,yticks=_get(kwargs.get('ytickses', None), i)
            ,yticklabels=_get(kwargs.get('yticklabelses', None), i)
            ,xlim=_get(kwargs.get('xlims', None), i)
            ,ylim=_get(kwargs.get('ylims', None), i)
            ,colors=_get(kwargs.get('colorses', None), i)
            ,zorder=_get(kwargs.get('zorders', None), i)
            ,title=_get(kwargs.get('titles', None), i)
            ,**kwargs)
    if kwargs.get('suptitle') is not None:
        fig.suptitle(kwargs.get('suptitle'), y = np.max(layout[:,1])+sh*1.5, verticalalignment='bottom')
    try:
        if kwargs.get('labels') is not None:
            plt.legend(handles=lineses[0], labels=kwargs.get('labels'), loc=kwargs.get('legend_loc', 3), fontsize=5, bbox_to_anchor=(np.max(layout, axis=0) - layout[i,:] + np.array([sw, sh])) // np.array([sw, sh]))
    except:
        pass
    if kwargs.get('savefig') is not None:
        _set(fig.savefig, kwargs.get('savefig'))
    if kwargs.get('show') is not None and not kwargs.get('show'):
        plt.close(fig)
    return fig
