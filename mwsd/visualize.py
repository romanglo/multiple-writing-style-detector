from __future__ import division
import logging

import matplotlib.pyplot as plt
import numpy as np


def visualize_zv(zv, show_plot=True, plot_saving_path=None):

    fig = plt.figure(num='ZV Distance')
    ax = fig.add_subplot(111)
    ax.plot(zv)

    ax.set_title('ZV Distance')
    ax.set_ylabel('ZV')
    ax.get_xaxis().set_visible(False)

    zv_max_index = np.argmax(zv)
    zv_max = zv[zv_max_index]

    ax.hlines(
        zv_max, 0, zv.shape[0], colors='r', linestyles='dashed', label='max')
    ax.text(1, zv_max + 0.05, "ZV max = {:.2f}".format(zv_max))

    ax.set_ylim(0, zv_max + 1)

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path, dpi=1000)
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
            plt.savefig(plot_saving_path, dpi=1000)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))

    if show_plot:
        plt.show()


def visualize_clustered_dzv(dzv,
                            clustering_result,
                            show_plot=True,
                            plot_saving_path=None):

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, num='Clustered DZV Distance')

    fig.suptitle('Clustered DZV Distance', fontsize=16)

    labels, distances, silhouette = clustering_result

    ax1.plot(labels)
    ax1.set_ylabel('Cluster')
    ax1.set_xlabel('Blocks')
    ax1.set_ylim(0, np.max(labels) + 1)
    ax1.text(
        0.5,
        0.01,
        'Silhouette Score: {:.4f}'.format(silhouette),
        verticalalignment='bottom',
        horizontalalignment='center',
        transform=ax1.transAxes,
        color='blue',
        fontsize=15)

    distances_colors = np.zeros(labels.shape[0], dtype=np.int)
    for unique in np.unique(labels):
        indexes = np.where(labels == unique)
        distances_colors[indexes] = (
            distances[indexes] * 255) / distances[indexes].max()

    clustered_dzv = np.zeros(dzv.shape, dtype=np.dtype((np.int32, (3, ))))
    for i in range(len(labels)):
        color_index = (labels[i] - 1) % 3
        for color in clustered_dzv[i, :]:
            color[color_index] = distances_colors[i]

    ax2.get_yaxis().set_visible(False)
    ax2.set_ylabel('Second Text')
    ax2.get_xaxis().set_visible(False)
    ax2.set_xlabel('First Text')
    plt.imshow(clustered_dzv)

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path, dpi=1000)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))

    if show_plot:
        plt.show()

    plt.show()


def visualize(zv,
              dzv,
              clustering_result,
              show_plot=True,
              plot_saving_path=None):
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

    ax1.hlines(
        zv_max, 0, zv.shape[0], colors='r', linestyles='dashed', label='max')
    ax1.text(1, zv_max + 0.05, "ZV max = {:.2f}".format(zv_max))

    ax1.set_ylim(0, zv_max + 1)

    ax2.imshow(dzv, interpolation='nearest', cmap=plt.cm.Blues)

    ax2.set_title('DZV Distance')
    ax2.set_ylabel('Second Text')
    ax2.set_xlabel('First Text')

    labels, distances, silhouette = clustering_result

    ax3.set_title('Clustered DZV Distance')
    ax3.plot(labels)
    ax3.set_ylabel('Cluster')
    ax3.set_xlabel('Blocks')
    ax3.set_ylim(0, np.max(labels) + 1)
    ax3.text(
        0.5,
        0.01,
        'Silhouette Score: {:.4f}'.format(silhouette),
        verticalalignment='bottom',
        horizontalalignment='center',
        transform=ax3.transAxes,
        color='blue',
        fontsize=15)

    distances_colors = np.zeros(labels.shape[0], dtype=np.int)
    for unique in np.unique(labels):
        indexes = np.where(labels == unique)
        distances_colors[indexes] = (
            distances[indexes] * 255) / distances[indexes].max()

    clustered_dzv = np.zeros(dzv.shape, dtype=np.dtype((np.int32, (3, ))))
    for i in range(len(labels)):
        color_index = (labels[i] - 1) % 3
        for color in clustered_dzv[i, :]:
            color[color_index] = distances_colors[i]

    ax4.set_title('Clustered DZV Distance')
    ax4.get_yaxis().set_visible(False)
    ax4.set_ylabel('Second Text')
    ax4.get_xaxis().set_visible(False)
    ax4.set_xlabel('First Text')
    plt.imshow(clustered_dzv)

    fig.tight_layout()

    if plot_saving_path is not None:
        try:
            plt.savefig(plot_saving_path, dpi=1000)
        except Exception:
            logging.error("Failed on try to save the plot in path: {}".format(
                plot_saving_path))

    if show_plot:
        plt.show()
