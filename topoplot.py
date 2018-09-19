import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

def sub_topoplot(ax, array, x_i=0, line_i=1, av_i=None, **kwargs):
    mpl.rcParams['axes.linewidth'] = 0.2
    ax_i = np.array([x_i, line_i, av_i])
    if array.ndim <= ax_i[pd.notnull(ax_i)].max():
        print('Error! There is no axis %d in "array".'%ax_i[pd.notnull(ax_i)].max())
        return
    nonuse_i = list(set(range(array.ndim)).difference(set(ax_i[pd.notnull(ax_i)])))
    if not all([array.shape[i] ==1 for i in nonuse_i]):
        print('Error! Size of axis %s must be 1.'%nonuse_i)
        return
    array1 = np.transpose(array, ax_i[pd.notnull(ax_i)].tolist()+nonuse_i)
    x = np.arange(array1.shape[0])
    lines = [0] * array.shape[line_i]
    colors = kwargs.get('color', [plt.get_cmap('tab10')(i%10) for i in range(len(lines))])
    for i in range(len(lines)):
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
    if kwargs.get('xlim') is not None:
        ax.set_xlim(kwargs.get('xlim'))
    if kwargs.get('ylim') is not None:
        ax.set_ylim(kwargs.get('ylim'))
    if kwargs.get('hlines') is not None:
        for hline in np.atleast_1d(kwargs.get('hlines')):
            [ax.arrow(ax.get_xlim()[0], hline_y, np.ptp(ax.get_xlim()), 0, lw=0.1, color='gray', ls='dashed', head_width=0, head_length=0) for hline_y in np.atleast_1d(hline)]
    if kwargs.get('vlines') is not None:
        for vline in np.atleast_1d(kwargs.get('vlines')):
            [ax.arrow(vline_x, ax.get_ylim()[0], 0, np.ptp(ax.get_ylim()), lw=0.1, color='gray', ls='dashed', head_width=0, head_length=0) for vline_x in np.atleast_1d(vline)]
    [tick.label.set_fontsize(2) for tick in ax.xaxis.get_major_ticks()]
    ax.tick_params(direction='in', length=1, width=0.2, pad=0.4)
    [tick.label.set_fontsize(2) for tick in ax.yaxis.get_major_ticks()]
    if kwargs.get('title') is not None:
        ax.set_title(kwargs.get('title'), fontsize=4)
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

def topoplot(list_of_array, layout='grid', sw=None, sh=None, subtopofunc=None, **kwargs):
    if isinstance(layout, str) and layout == 'grid':
        grid_size = int(np.ceil(np.sqrt(len(list_of_array))))
        layout = np.stack((
            np.tile(np.arange(grid_size),grid_size)
            ,np.tile(np.arange(grid_size), (grid_size,1)).T.flatten()
        )).T[:len(list_of_array)+1,:]
        sw = 1/grid_size
    layout = (layout - np.min(layout, axis=0)) / np.ptp(layout, axis=0)
    if sw is None or sh is None:
        sw_, sh_ = _find_best_swh(layout, kwargs.get('wh', 1))
    if sw is None:
        sw = sw_
    if sh is None:
        sh = sh_
    if subtopofunc is None:
        subtopofunc = sub_topoplot
    fig = plt.figure(dpi=kwargs.get('dpi', 320), figsize=kwargs.get('figsize', (6,4)))
    lineses = [0] * len(list_of_array)
    for i, array in enumerate(list_of_array):
        ax = fig.add_axes(list(layout[i,:]) + [sw, sh])
        lineses[i] = subtopofunc(ax, array
            ,xlim=kwargs.get('xlims', [None] * len(list_of_array))[i]
            ,ylim=kwargs.get('ylims', [None] * len(list_of_array))[i]
            ,title=kwargs.get('titles', [None] * len(list_of_array))[i]
            ,**kwargs)
    if kwargs.get('suptitle') is not None:
        fig.suptitle(kwargs.get('suptitle'), y = np.max(layout[:,1])+sh*1.2)
    try:
        if kwargs.get('labels') is not None:
            plt.legend(handles=lineses[0], labels=kwargs.get('labels'), fontsize='x-small')
    except:
        pass
    return fig
