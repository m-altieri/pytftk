import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt
from colorama import Fore, Style

import pytftk.nn


class TBManager:
    def __init__(
        self,
        tb_folder,
        run_name=None,
        tmp=False,
        interval=None,
        enabled=True,
        verbose=False,
        port=None,
    ):
        self.tb_folder = tb_folder
        self.run_name = run_name
        self.tmp = tmp
        DEFAULT_INTERVAL = 100
        self.interval = DEFAULT_INTERVAL if interval is None else interval
        DEFAULT_PORT = 6006
        self.port = DEFAULT_PORT if port is None else port
        self.enabled = enabled
        self.force_write = False
        self.verbose = verbose
        self.scalars = {}
        self.images = {}
        self.histograms = {}
        self.texts = {}

        # set tensorboard path and create folders
        self.tb_path = None
        if not os.path.exists(self.tb_folder):
            os.makedirs(self.tb_folder)
        if not self.tmp:
            self.tb_path = os.path.join(
                self.tb_folder,
                self.run_name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        else:
            self.tb_path = os.path.join(self.tb_folder, "tmp")

        # remove existing files in folder if it exists
        if os.path.exists(self.tb_path):
            for filename in os.listdir(self.tb_path):
                os.remove(os.path.join(self.tb_path, filename))

        # start tensorboard instance
        self.tb_instance = None

        # initialize counter variable
        self.counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        # initialize tf file writer
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)

    def run(self):
        self.tb_instance = tb.program.TensorBoard()
        self.tb_instance.configure(logdir=self.tb_path, port=self.port)
        tb_url = self.tb_instance.launch()
        print(f"[INFO] TensorBoard instance launched at {tb_url}")

    def is_hot(self):
        if self.force_write:
            if self.enabled:
                if self.verbose:
                    print("[DEBUG] TBManager is hot because it is forced.")
            else:
                print(
                    "[WARN] You are forcing TBManager to stay hot but it is not enabled."
                )
        return self.enabled and (self.counter % 100 == 0 or self.force_write)

    def step(self):
        if self.enabled:
            self.counter.assign_add(tf.constant(1, dtype=tf.int64))

    def force(self, value, /):
        """Make is_hot() return True until you call force(False), unless
        the TBManager is disabled (as if interval is temporarily 1).
        Note: this can have unexpected behaviour. Use only for debugging.

        :param value: Whether to force the TBManager to stay hot.
        :type value: bool
        """
        self.force_write = value

    def scalar(self, name, data):
        if self.enabled:
            if name not in self.scalars:
                self.scalars[name] = {"step": 0}

            logging_step = self.scalars[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing scalar with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )

            with self.tb_writer.as_default():
                tf.summary.scalar(name, data, step=logging_step)

            self.scalars[name]["step"] = logging_step + 1

    def image(self, name, data, minmax=False, highlight_mask=None):
        """Visualize a tf.Tensor as an image in TensorBoard.

        Args:
            name (str): The name of the entry in TensorBoard.
            data (tf.Tensor): A tf.Tensor of shape [B,X,Y] to visualize as an
            image in TensorBoard. For example, X and Y may be T and N.
            minmax (bool, optional): Whether to normalize the pixel values between
            0 and 1. Defaults to False.
            highlight_mask (tf.Tensor, optional): A tf.Tensor of shape [B,X,Y]
            representing the elements of data to highlight. Defaults to None.
        """
        if self.enabled:
            if name not in self.images:
                self.images[name] = {"step": 0}

            logging_step = self.images[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing image with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )
            # if the channel (color) axis is missing, add it with dim 1 (greyscale)
            data = tf.cond(
                tf.math.equal(tf.rank(data), tf.constant(3)),
                lambda: tf.expand_dims(data, -1),
                lambda: data,
            )
            # if minmax, apply minmax normalization to the image
            data = tf.cond(
                tf.constant(minmax),
                lambda: pytftk.nn.minmax(data),
                lambda: data,
            )

            # if highlight_mask, tile the channel value to 3 channels (RGB)
            # and set the R and B channels to 0 for indices in which mask is 1
            # (highlighting with a green tint the masked elements)
            if highlight_mask is not None:
                # data: [B,T,N,1], highlight_mask: [B,T,N]
                data = tf.tile(data, [1, 1, 1, 3])  # assumes data has rank 4
                highlight_mask = tf.cast(highlight_mask, tf.bool)

                # 0=R, 2=B removed (green tint)
                indexing_tensor = tf.constant([[0], [2]])

                data = tf.where(
                    highlight_mask,
                    # tensor like data but with R and B channels zero'd
                    tf.tensor_scatter_nd_update(
                        tf.transpose(data, [3, 0, 1, 2]),  # trick (easier to index)
                        indexing_tensor,  # select channels for all pixels
                        tf.stack(
                            [
                                tf.zeros_like(highlight_mask, dtype=data.dtype),
                                tf.zeros_like(highlight_mask, dtype=data.dtype),
                            ],
                            axis=0,
                        ),  # set their values to 0 (leave only green)
                    ),
                    # original data tensor
                    tf.transpose(data, [3, 0, 1, 2]),  # trick (easier to index)
                )
                data = tf.transpose(data, [1, 2, 3, 0])  # undo trick

            with self.tb_writer.as_default():
                tf.summary.image(name, data, step=logging_step)

            self.images[name]["step"] = logging_step + 1

    def histogram(self, name, data):
        if self.enabled:
            if name not in self.histograms:
                self.histograms[name] = {"step": 0}

            logging_step = self.histograms[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing histogram with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )

            with self.tb_writer.as_default():
                tf.summary.histogram(name, data, step=logging_step)

            self.histograms[name]["step"] = logging_step + 1

    def text(self, name, data):
        if self.enabled:
            if name not in self.texts:
                self.texts[name] = {"step": 0}

            logging_step = self.texts[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing text with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )

            with self.tb_writer.as_default():
                tf.summary.text(name, data, step=logging_step)

            self.texts[name]["step"] = logging_step + 1


class Logbook:
    """Keeps a dict of lists of lists:
    {'m1': [[...], [...], ..., [...]], ..., 'mn': [[...], [...], ..., [...]]}
    """

    def __init__(self):
        super().__init__()
        self.metrics = {}
        # self.figure = plt.figure()

    def register(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = [[]]

        self.metrics[name][-1].append(value)

    def get(self, name, overall=False):
        if name not in self.metrics:
            print(
                f"{Fore.YELLOW}[WARN] Can't get metric {Style.BRIGHT}{name}"
                + f"{Style.NORMAL}: not in logbook. Logbook contains: "
                + f"{Style.BRIGHT}{[k for k in self.metrics]}{Style.RESET_ALL}"
            )
        else:
            return self.metrics[name] if overall else self.metrics[name][-1]

    def count(self, name, overall=False):
        if name not in self.metrics:
            print(
                f"{Fore.YELLOW}[WARN] Can't count metric {Style.BRIGHT}{name}"
                + f"{Style.NORMAL}: not in logbook. Logbook contains: "
                + f"{Style.BRIGHT}{[k for k in self.metrics]}{Style.RESET_ALL}"
            )
        else:
            return (
                np.size(self.metrics[name]) if overall else len(self.metrics[name][-1])
            )

    def new(self):
        for name in self.metrics:
            self.metrics[name].append([])

    def save_plot(
        self,
        path,
        names=None,  # names of the metrics to plot. if None, plot them all. None, string or list
        title=None,
        color="black",  # str or dict. if dict, the color of each metric
        linewidth=1,
        alpha=1.0,
        pad_left=0.2,
        pad_bottom=0.2,
        xlabel="Steps",
        ylabel="Value",
        xticks_every=None,
        multiplot=False,  # plot all selected metrics in the same plot
    ):
        # convert `names` into a list
        if names is None:
            names = [k for k in self.metrics]
        if type(names) is str:
            names = [names]

        # print warning if one or more metrics is not present in the logbook
        not_present = [m for m in names if m not in self.metrics]
        if len(not_present) > 0:
            print(
                f"{Fore.YELLOW}[WARN] Can't plot these metrics: {Style.BRIGHT}"
                + f"{', '.join(not_present)}{Style.NORMAL}, as they are not "
                + f"in logbook. Logbook contains: {Style.BRIGHT}"
                + f"{[k for k in self.metrics]}.{Style.RESET_ALL}"
            )

        # print warning if one or more metrics has no registered values
        empty = [m for m in names if self.count(m) == 0]
        if len(empty) > 0:
            print(
                f"{Fore.YELLOW}[WARN] Metrics {Style.BRIGHT}{', '.join(empty)}"
                + f"{Style.NORMAL} have no registered values and will not be "
                + f"saved.{Style.RESET_ALL}"
            )

        # redefine names to only contain present metrics
        names = [m for m in names if m not in not_present and m not in empty]
        if len(names) == 0:
            print(
                f"{Fore.YELLOW}[WARN] All provided metrics are either not "
                + f"present in the logbook or are empty. "
                + f"{self.save_plot.__name__} will return now.{Style.RESET_ALL}"
            )
            return

        for metric in names:
            if not multiplot:  # initialize a new plot for each metric
                plt.clf()
                plt.figure(metric)
            plt.plot(  # plot the current metric
                np.arange(self.count(metric)),
                np.array(self.get(metric)),
                linewidth=(
                    linewidth if type(linewidth) is not dict else linewidth[metric]
                ),
                color=color if type(color) is not dict else color[metric],
                alpha=alpha if type(alpha) is not dict else alpha[metric],
                label=metric,
            )
            if not multiplot:  # configure the plot for the current metric
                plt.title(
                    title or "" if type(title) is not dict else title[metric],
                    fontdict={"fontsize": 24},
                )
                plt.xticks(
                    np.arange(
                        0,
                        self.count(metric),
                        xticks_every or max(self.count(metric) // 10, 1),
                    ),
                    fontsize=24,
                    rotation=60,
                )
                plt.yticks(fontsize=24)
                plt.xlabel(xlabel, fontsize=24)
                plt.ylabel(metric, fontsize=24)
                plt.ylim(bottom=0)
                plt.grid(True, axis="y")
                plt.gcf().subplots_adjust(left=pad_left, bottom=pad_bottom)
                plt.savefig(os.path.join(path, metric + ".png"))
                plt.savefig(os.path.join(path, metric + ".pdf"))
        if multiplot:  # configure the plot globally for all metrics
            plt.title(title or "", fontdict={"fontsize": 24})
            plt.xticks(
                np.arange(
                    0,
                    self.count(names[0]),
                    xticks_every or max(self.count(metric) // 10, 1),
                ),
                fontsize=24,
                rotation=60,
            )
            plt.yticks(fontsize=24)
            plt.xlabel(xlabel, fontsize=24)
            plt.ylabel(ylabel, fontsize=24)
            plt.ylim(bottom=0)
            plt.grid(True, axis="y")
            plt.legend()
            plt.gcf().subplots_adjust(left=pad_left, bottom=pad_bottom)
            plt.savefig(os.path.join(path, "_".join(names) + ".png"))
            plt.savefig(os.path.join(path, "_".join(names) + ".pdf"))

        for metric in names:
            values = np.array(self.get(metric))
            df = pd.DataFrame(values)
            save_path = os.path.join(path, metric + ".csv")
            df.to_csv(save_path)
            print(
                f"{Fore.CYAN}[INFO] Saved {Style.BRIGHT}{metric}{Style.NORMAL} "
                + f"to {Style.BRIGHT}{save_path}{Style.NORMAL}.{Fore.RESET}"
            )
