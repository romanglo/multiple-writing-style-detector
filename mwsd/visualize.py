import numpy as np
import matplotlib.pyplot as plt


def visualize_zv(zv, show_plot=True, plot_saving_path=None):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(zv)

    ax.set_title("ZV Distance")
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

    if plot_saving_path is not None:
        plt.savefig(plot_saving_path)

    if show_plot:
        plt.show()
