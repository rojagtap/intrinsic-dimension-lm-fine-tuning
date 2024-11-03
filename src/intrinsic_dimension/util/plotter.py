import os

import matplotlib.pyplot as plt
from torchviz import make_dot

from .constants import DEVICE


def close_plot():
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')


def plot_results(baseline, dints, performance, basedir, name, xlabel=None, ylabel=None, title=None, show_dint90=False):
    bubble_size = 100
    fig, ax = plt.subplots()

    ax.axhline(y=baseline, linestyle='-' + '-' * bool(not show_dint90), color='black', linewidth=1, label='baseline')
    if show_dint90:
        ax.axhline(y=0.9 * baseline, linestyle='--', color='black', linewidth=1, label='90% baseline')

    # Plot DID performance
    ax.scatter(dints, performance["did"], s=bubble_size, alpha=0.5, edgecolors='blue', color='darkblue', label='DID Performance')
    ax.plot(dints, performance["did"], linestyle='-', color='blue', linewidth=1)

    # Plot SAID performance
    ax.scatter(dints, performance["said"], s=bubble_size, alpha=0.5, edgecolors='green', color='darkgreen', label='SAID Performance')
    ax.plot(dints, performance["said"], linestyle='-', color='green', linewidth=1)

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.legend(loc="best")

    # save plot
    os.makedirs(os.path.join(basedir, "plot"), exist_ok=True)

    plt.savefig(os.path.join(basedir, "plot/{}.png".format(name)))
    close_plot()


def plot_model(model, name, basedir, sample):
    # save plot
    os.makedirs(os.path.join(basedir, "plot"), exist_ok=True)

    model.to(DEVICE)
    sample.to(DEVICE)
    make_dot(
        model(input_ids=sample["input_ids"],
              attention_mask=sample["attention_mask"],
              token_type_ids=sample["token_type_ids"],
              labels=sample["labels"])["loss"],
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True
    ).render(outfile=f"{basedir}/plot/{name}.png")


def plot_intrinsic_dimensions(values, basedir, scale):
    plt.figure(figsize=(10, 6))
    plt.plot(values, marker='o', linestyle='None')
    plt.yscale(scale)
    plt.xlabel('Index')
    plt.ylabel('Intrinsic Dimensionality (d)')
    plt.title('Intrinsic Dimensionality (d) Values Generated')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(basedir, "plot/intrinsic-dimension-gen.png"))
    close_plot()
