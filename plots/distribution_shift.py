# -*- coding: utf-8 -*-
"""
Created on 2020/11/12 9:04

@author: John_Fengz
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
config = {
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Helvetica'],
}
rcParams.update(config)


def get_kde(file_path):
    data = pd.read_excel(file_path)
    features = np.array(data[[3]])
    features = scale(features)
    kde_ns = KernelDensity(kernel='gaussian', bandwidth=0.15)
    kde_ns.fit(features[:200])

    kde_ds = KernelDensity(kernel='gaussian', bandwidth=0.15)
    kde_ds.fit(features[-200:])
    return kde_ns, kde_ds


folder = '../data/'
kde_ns1, kde_ds1 = get_kde(folder + 'xBearing1_1.xlsx')
kde_ns2, kde_ds2 = get_kde(folder + 'xBearing1_2.xlsx')
kde_ns3, kde_ds3 = get_kde(folder + 'xBearing1_3.xlsx')
kde_ns4, kde_ds4 = get_kde(folder + 'xBearing1_4.xlsx')
kde_ns5, kde_ds5 = get_kde(folder + 'xBearing1_5.xlsx')
kde_ns6, kde_ds6 = get_kde(folder + 'xBearing1_6.xlsx')
kde_ns7, kde_ds7 = get_kde(folder + 'xBearing1_7.xlsx')

x_plot_ns = np.linspace(-1, 1, 50)[:, np.newaxis]
dens_ns1 = np.exp(kde_ns1.score_samples(x_plot_ns))
dens_ns2 = np.exp(kde_ns2.score_samples(x_plot_ns))
dens_ns3 = np.exp(kde_ns3.score_samples(x_plot_ns))
dens_ns4 = np.exp(kde_ns4.score_samples(x_plot_ns))
dens_ns5 = np.exp(kde_ns5.score_samples(x_plot_ns))
dens_ns6 = np.exp(kde_ns6.score_samples(x_plot_ns))
dens_ns7 = np.exp(kde_ns7.score_samples(x_plot_ns))
dens_ns = [dens_ns1, dens_ns2, dens_ns3, dens_ns4, dens_ns5, dens_ns6, dens_ns7]

x_plot_ds = np.linspace(-1.5, 4, 50)[:, np.newaxis]
dens_ds1 = np.exp(kde_ds1.score_samples(x_plot_ds))
dens_ds2 = np.exp(kde_ds2.score_samples(x_plot_ds))
dens_ds3 = np.exp(kde_ds3.score_samples(x_plot_ds))
dens_ds4 = np.exp(kde_ds4.score_samples(x_plot_ds))
dens_ds5 = np.exp(kde_ds5.score_samples(x_plot_ds))
dens_ds6 = np.exp(kde_ds6.score_samples(x_plot_ds))
dens_ds7 = np.exp(kde_ds7.score_samples(x_plot_ds))
dens_ds = [dens_ds1, dens_ds2, dens_ds3, dens_ds4, dens_ds5, dens_ds6, dens_ds7]

cmap = plt.get_cmap('viridis')
colors = [cmap(x) for x in np.linspace(0, 1, 7)]

plt.figure(figsize=(7.5, 3.5))
ax1 = plt.subplot(121)
for i, dn in enumerate(dens_ns):
    ax1.plot(x_plot_ns[:, 0], dn, color=colors[i], linewidth=1, label='b'+str(i+1))
ax1.legend(labelspacing=0, borderaxespad=0.2, fontsize=10)
ax1.set_ylabel('$P(RMS_{ns})$')
ax1.set_xlabel('$RMS_{ns}$')
plt.margins(x=0, y=0)
ax1.set_title('(a)', y=-0.3)

ax2 = plt.subplot(122)
for i, dd in enumerate(dens_ds):
    ax2.plot(x_plot_ds[:, 0], dd, color=colors[i], linewidth=1, label='b'+str(i+1))
ax2.legend(labelspacing=0, borderaxespad=0.2, fontsize=10)
ax2.set_ylabel('$P(RMS_{ds})$')
ax2.set_xlabel('$P(RMS_{ds})$')
plt.margins(x=0, y=0)
ax2.set_title('(b)', y=-0.3)

plt.tight_layout()
plt.savefig('distribution_shift.png', dpi=800, pad_inches=0)
plt.savefig('distribution_shift.pdf', dpi=800, pad_inches=0)
plt.show()
