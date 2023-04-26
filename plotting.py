import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

def plot_state_space(data, x=0, y=1, ax=None, palette='viridis', lw=2):
    """
    plot relationship between contribution of two components over time
    param: x,y int, component number

    """
    if ax is None:
        fig, ax = plt.subplots(1,1)

    x_data = data[:,x]
    y_data = data[:,y]

    # line segments
    points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, len(x_data)-1)
    lc = LineCollection(segments, cmap=palette, norm=norm)
    lc.set_array(np.arange(len(x_data)))
    lc.set_linewidth(lw)

    # plot
    ax.scatter(x_data, y_data, c=np.arange(4500), s=0)
    ax.add_collection(lc)

    # label
    ax.set_xlabel('comp {:}'.format(x), fontsize=15)
    ax.set_ylabel('comp {:}'.format(y), fontsize=15)
    sns.despine(bottom=True, left=True)
    ax.tick_params(axis='both', labelbottom=False, bottom=False, labelleft=False, left=False)

def plot_pm_scatter(axis, frame, data):
    x = data[frame, ::2]
    y = data[frame, 1::2]

    axis.scatter(y, -x, edgecolor='grey', facecolor='#d36084', s=100)

    if frame > 2:
        pre_x = data[frame-2, ::2]
        pre_y = data[frame-2, 1::2]
        axis.scatter(pre_y, -pre_x, edgecolor='grey', facecolor='#d36084', s=90, alpha=0.8, zorder=0)
    if frame > 4:
        pre_x = data[frame-4, ::2]
        pre_y = data[frame-4, 1::2]
        axis.scatter(pre_y, -pre_x, edgecolor='grey', facecolor='#d36084', s=80, alpha=0.5, zorder=0)
    if frame > 8:
        pre_x = data[frame-8, ::2]
        pre_y = data[frame-8, 1::2]
        axis.scatter(pre_y, -pre_x, edgecolor='grey', facecolor='#d36084', s=70, alpha=0.2, zorder=0)


def plot_principal_movement(reconstructed_movement, weights, outpath):

    num_frames = np.shape(reconstructed_movement)[0]

    for frame in np.arange(num_frames):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,8), gridspec_kw={'height_ratios': [2.5, 1]})

        plot_pm_scatter(ax1, frame, reconstructed_movement)

        norm = plt.Normalize(np.min(weights), np.max(weights))
        ax2.scatter(frame, weights[frame], c=weights[frame], norm=norm,  cmap='pink', edgecolors='grey')
        if frame > 2:
            ax2.scatter(frame-1, weights[frame-1], alpha=.5, c=weights[frame-1], norm=norm,  cmap='pink', zorder=0, edgecolors='grey')
        if frame > 4:
            ax2.scatter(frame-2, weights[frame-2], alpha=.5, c=weights[frame-2], norm=norm,  cmap='pink', zorder=0, edgecolors='grey')
        if frame > 8:
            ax2.scatter(frame-2, weights[frame-2], alpha=.5, c=weights[frame-2], norm=norm,  cmap='pink', zorder=0, edgecolors='grey')

        ax1.set_ylim(np.min(-reconstructed_movement[:,::2])*1.05, np.max(-reconstructed_movement[:,::2])*1.05)
        ax1.set_xlim(np.min(reconstructed_movement[:,1::2])*1.05, np.max(reconstructed_movement[:,1::2])*1.05)

        ax2.set_ylim(-6, 6)
        ax2.set_xlim(0,100)

        sns.despine(ax=ax1, bottom=True, left=True)
        ax1.tick_params(axis='both', labelbottom=False, bottom=False, labelleft=False, left=False)
        ax2.set_ylabel('component weight', fontsize=15)
        ax2.set_xlabel('time', fontsize=15)
        ax2.tick_params(axis='both', labelsize=15)

        plt.tight_layout()
        plt.savefig('{:}/frame_{:03d}.png'.format(outpath, frame), transparent=False)
        plt.close()


def plot_all_movements(recon_movements, outpath):

    num_cols = 5
    num_plots = len(recon_movements)
    num_rows = int(np.ceil(num_plots / num_cols))

    num_frames = np.shape(recon_movements[0])[0]
    for frame in np.arange(num_frames):

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,20), sharex=True, sharey=True)
        axes = axes.reshape(-1)

        for i in np.arange(num_plots):

            plot_pm_scatter(axes[i], frame, recon_movements[i])

            axes[i].set_ylim(np.min(-recon_movements[i][:,::2])*1.2, np.max(-recon_movements[i][:,::2])*1.2)
            axes[i].set_xlim(np.min(recon_movements[i][:,1::2])*1.2, np.max(recon_movements[i][:,1::2])*1.2)

            sns.despine(ax=axes[i], bottom=False, left=False, top=False, right=False)
            axes[i].tick_params(axis='both', labelbottom=False, bottom=False, labelleft=False, left=False)

        plt.tight_layout()
        plt.savefig('{:}/all_components_frame_{:03d}.png'.format(outpath, frame), transparent=False, dpi=300)
        plt.close()

def plot_spectra(spectrogram, timeseries, n_freqs = 10, n_comps = 10, outpath='.'):

    fig, ax = plt.subplots(n_comps, 1, figsize=(5,n_comps), sharex = True)

    for n in np.arange(n_comps):

        ts = timeseries[n,:]

        spec = spectrogram[n*n_freqs:(n+1)*n_freqs,:]

        ax[n].imshow(spec, aspect='auto', cmap='Greys')
        ax[n].invert_yaxis()
        ax[n].set_title('mode {:}'.format(n+1), loc='left', fontsize=13)
        ax[n].set_yticks([0, n_freqs])
        ax[n].set_yticklabels(['low', 'high'])

        ax1 = ax[n].twinx()
        ax1.plot(ts, c='black', lw=0.5)
        ax1.set_yticklabels([])

    ax[-1].set_xlabel('time', fontsize=13)
    plt.tight_layout()
    plt.savefig('{:}/spectra.png'.format(outpath), transparent=False)
    plt.show()
