import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def visualize_zv(zv, show_plot=True, plot_saving_path=None):

    fig = plt.figure(num='ZV Distance')
    ax = fig.add_subplot(111)
    line, = ax.plot(zv)

    ax.set_title('ZV Distance')
    ax.set_ylabel('ZV')
    ax.get_xaxis().set_visible(False)

    zv_max_index = np.argmax(zv)
    zv_max = zv[zv_max_index]

    ax.annotate(
        "ZV max = {:.2f}".format(zv_max),
        xy=(zv_max_index, zv_max),
        xytext=(zv_max_index - 5, zv_max + 5),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )

    ax.set_ylim(0, zv_max + 10)

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))

    if show_plot:
        plt.show()


def visualize_dzv(dzv, show_plot=True, plot_saving_path=None):
    fig, ax = plt.subplots()

    ax.imshow(dzv, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set_title('DZV Distance')
    ax.set_ylabel('Second Text')
    ax.set_xlabel('First Text')

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))

    if show_plot:
        plt.show()


def visualize_clustered_dzv(dzv,
                            medoids,
                            show_plot=True,
                            plot_saving_path=None):

    dzv_copy = np.copy(dzv)
    dzv_copy *= 1.0 / (dzv_copy.max() + 0.01)

    res = np.zeros(dzv.shape[0], dtype=np.int)

    for i, row in enumerate(dzv):
        for j, medoid in enumerate(medoids):
            found = False
            for element in medoid.elements:
                if np.array_equal(row, element):
                    found = True
                    break
            if found:
                res[i] = j + 1
                dzv_copy[i, :] += j
                break

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, num='Clustered DZV Distance')

    line, = ax1.plot(res)

    ax1.set_ylabel('Cluster')
    ax1.set_xlabel('Blocks')

    ax1.set_ylim(0, np.max(res) + 1)

    cmap, norm = mcolors.from_levels_and_colors([0, 1, 2, 3, 4], [
        'blue',
        'yellow',
        'red',
        'green',
    ])

    ax2.pcolor(dzv_copy, cmap=cmap, norm=norm)

    ax2.invert_yaxis()
    ax2.set_ylabel('Second Text')
    ax2.set_xlabel('First Text')

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))
    if show_plot:
        plt.show()

    plt.show()


def visualize(zv, dzv, medoids, show_plot=True, plot_saving_path=None):
    fig, (ax_row, ax_col) = plt.subplots(
        nrows=2, ncols=2, num='Algorithm Results')

    (ax1, ax2) = ax_row
    (ax3, ax4) = ax_col

    line, = ax1.plot(zv)

    ax1.set_title('ZV Distance')
    ax1.set_ylabel('ZV')
    ax1.get_xaxis().set_visible(False)

    zv_max_index = np.argmax(zv)
    zv_max = zv[zv_max_index]

    ax1.annotate(
        "ZV max = {:.2f}".format(zv_max),
        xy=(zv_max_index, zv_max),
        xytext=(zv_max_index - 5, zv_max + 5),
        arrowprops=dict(facecolor='black', shrink=0.05),
    )

    ax1.set_ylim(0, zv_max + 10)

    ax2.imshow(dzv, interpolation='nearest', cmap=plt.cm.Blues)

    ax2.set_title('DZV Distance')
    ax2.set_ylabel('Second Text')
    ax2.set_xlabel('First Text')

    dzv_copy = np.copy(dzv)
    dzv_copy *= 1.0 / (dzv_copy.max() + 0.01)

    res = np.zeros(dzv.shape[0], dtype=np.int)

    for i, row in enumerate(dzv):
        for j, medoid in enumerate(medoids):
            found = False
            for element in medoid.elements:
                if np.array_equal(row, element):
                    found = True
                    break
            if found:
                res[i] = j + 1
                dzv_copy[i, :] += j
                break

    line, = ax3.plot(res)

    ax3.set_ylabel('Cluster')
    ax3.set_xlabel('Blocks')
    ax3.set_title('Clustered DZV Distance')

    ax3.set_ylim(0, np.max(res) + 1)

    cmap, norm = mcolors.from_levels_and_colors([0, 1, 2, 3, 4], [
        'blue',
        'yellow',
        'red',
        'green',
    ])

    ax4.pcolor(dzv_copy, cmap=cmap, norm=norm)

    ax4.invert_yaxis()
    ax4.set_ylabel('Second Text')
    ax4.set_xlabel('First Text')
    ax4.set_title('Clustered DZV Distance')

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))

    if show_plot:
        plt.show()
