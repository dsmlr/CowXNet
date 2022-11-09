import numpy as np
import matplotlib.pyplot as plt

def compute_cumulative_estrus_vector(estrus_result):
    
    cumulative_estrus = []
    n_intervals = 0

    for i in range(len(estrus_result)):

        if i == 0:
            cumulative_value = estrus_result[i]
            cumulative_estrus.append(cumulative_value)
            continue

        if estrus_result[i] == 1:
            cumulative_value += 1
        else:
            if cumulative_value != 0:
                n_intervals += 1
            cumulative_value = 0

        cumulative_estrus.append(cumulative_value)

    cumulative_estrus = np.array(cumulative_estrus)

    if 1 in cumulative_estrus:
        avg_time_cont = np.average(cumulative_estrus[np.nonzero(cumulative_estrus)])
        min_time_cont = np.min(cumulative_estrus[np.nonzero(cumulative_estrus)])
        max_time_cont = np.max(cumulative_estrus[np.nonzero(cumulative_estrus)])
    else:
        avg_time_cont = 'None'
        min_time_cont = 'None'
        max_time_cont = 'None'

    return_obj = {
        'cumulative_estrus': cumulative_estrus,
        'avg_time_cont': avg_time_cont,
        'min_time_cont': min_time_cont,
        'max_time_cont': max_time_cont,
        'n_intervals': n_intervals
    }
    
    return return_obj

def find_estrus_intervals(cumulative_estrus):
    estrus_intervals = []
    interval_i = []
    for i, v in enumerate(cumulative_estrus):
        if (v > 0) and (i != len(cumulative_estrus) - 1):
            interval_i.append(i)
        else:
            if len(interval_i) != 0:
                estrus_intervals.append(interval_i)
            interval_i = []

    nonestrus_intervals = []
    interval_i = []
    for i, v in enumerate(cumulative_estrus):
        if (v == 0) and (i != len(cumulative_estrus) - 1):
            interval_i.append(i)
        else:
            if len(interval_i) != 0:
                nonestrus_intervals.append(interval_i)
            interval_i = []
    
    return {
        'estrus_intervals': estrus_intervals,
        'nonestrus_intervals': nonestrus_intervals
    }

def create_labels(intervals, cumulative_estrus_pd):
    gt_labels = []
    pd_labels = []

    for i, interval in enumerate(intervals['estrus_intervals']):
        l = 1 if sum(cumulative_estrus_pd[interval]) > 0 else 0
        pd_labels.append(l)
        gt_labels.append(1)

    for i, interval in enumerate(intervals['nonestrus_intervals']):
        l = 1 if sum(cumulative_estrus_pd[interval]) > 0 else 0
        pd_labels.append(l)
        gt_labels.append(0)
        
    return {
        'gt': gt_labels,
        'pd': pd_labels
    }

def create_cumulative_estrus_graph(params, title=''):
    fig = plt.figure()
    fig.set_size_inches(15, 3)
    plt.plot(params['cumulative_estrus'], color='black', linewidth=1.5)
    plt.margins(x=0.01)
    plt.title(title, fontsize=20)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.grid(True)

    text_sum = r'$T_{average}= %.2f$ sec.%s$T_{min}= %d$ sec.%s$T_{max}= %d$ sec.%s$N_{interval}= %d$' % (
        params['avg_time_cont'],
        '\n',
        params['min_time_cont'],
        '\n',
        params['max_time_cont'],
        '\n',
        params['n_intervals'])
    plt.text(params['time'] - (params['time'] * 0.14), max(params['cumulative_estrus']) - max(params['cumulative_estrus']) * 0.23, text_sum, size=18,
             ha="left", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )

    return fig