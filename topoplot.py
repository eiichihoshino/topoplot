import numpy as np
import matplotlib.pyplot as plt
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
    array1 = np.transpose(array, ax_i[pd.notnull(ax_i)].tolist()+nonuse_i) * my
    x = np.arange(array1.shape[0])*mx+dx
    lines = [0] * array.shape[line_i]
    for i in range(array.shape[line_i]):
        if array1.ndim > 2 and array1.shape[2] > 1:
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y = np.nanmean(array1[:,i,], axis=1)
            y_sd = np.nanstd(array1[:,i,], axis=1)
            ax.fill_between(
                x, y+dy-y_sd, y+dy+y_sd
                ,facecolor=kwargs.get('color')[i], alpha=0.1)
        else:
            y = array1[:,i]
        lines[i] = ax.plot(
            x
            ,y+dy
            ,linewidth=0.3
            ,c=kwargs.get('color')[i]
        )[0]

    xlim = (dx, array1.shape[0]*mx+dx)
    ylim = (dy-my*scale/float(2), dy+my*scale/float(2))
    ax.fill_between(
                    x = x, y1=ylim[0], y2=ylim[1]
                    ,where=[False]*100+[True]*kwargs.get('stim_duration')+[False]*(array1.shape[0]-100-kwargs.get('stim_duration'))
                    ,facecolor='gray', alpha=0.2)
    plt.arrow(xlim[0], ylim[0], xlim[1] - xlim[0], 0, lw=0.1, color="k", head_width=0.2)
    plt.arrow(xlim[0], ylim[0], 0, ylim[1]-ylim[0], lw=0.1, color="k", head_width=0.2)
    if kwargs.get('text'):
        plt.text(xlim[0]+(xlim[1]-xlim[0])/2, ylim[1],  kwargs.get('text'), fontsize=4, horizontalalignment='center')
    return lines

def topoplot(list_of_array, layout='grid', mx=0.04, my=40, subtopofunc=None, **kwargs):
    fig = plt.figure(dpi=320)
    ax = plt.subplot(111)
    if subtopofunc is None:
        subtopofunc = sub_topoplot
    lineses = [0] * len(list_of_array)
    if kwargs.get('texts') is None:
        texts = [None] * len(list_of_array)
    else:
        texts = kwargs.get('texts')
    for i, array in enumerate(list_of_array):
        lineses[i] = subtopofunc(ax, array, mx=mx, my=my, dx=layout[i, 0], dy=layout[i, 1], **kwargs)
    try:
        plt.legend(lineses[0], fontsize='x-small', labels=kwargs.get('labels'))
    except:
        pass
    return ax, lineses
