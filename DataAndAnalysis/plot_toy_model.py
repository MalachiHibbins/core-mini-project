#!/usr/bin/env python3
"""
Plot the stages stored in ToyModelData.txt as colour maps.
Usage: python3 plot_toy_model.py [path/to/ToyModelData.txt]
"""
import sys
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def parse_toymodel_file(path):
    steps = []
    with open(path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    current_grid = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # skip separator lines consisting mostly of '='
        if set(ln) <= set('='):
            continue
        # Step header
        if ln.lower().startswith('step'):
            # if we have a grid accumulated, save it
            if current_grid:
                steps.append(current_grid)
                current_grid = []
            continue
        # otherwise it's a grid row like: "X . X . X X"
        tokens = ln.split()
        # accept only rows with characters of interest
        if all(t in {'.','T','B','X'} for t in tokens):
            current_grid.append(tokens)
    # append last grid
    if current_grid:
        steps.append(current_grid)

    # convert to numpy arrays with integer codes: .->0, T->1, B->2, X->3
    char_to_int = {'.':0, 'T':1, 'B':2, 'X':3}
    arrays = []
    for g in steps:
        arr = np.array([[char_to_int.get(c, 0) for c in row] for row in g], dtype=int)
        arrays.append(arr)
    return arrays


def plot_steps(arrays, out_path=None):
    S = len(arrays)
    if S == 0:
        raise SystemExit('No steps found in file')

    # layout
    ncols = min(4, S)
    nrows = ceil(S / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows * ncols == 1:
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        axs = axs.reshape(nrows, ncols)

    # define discrete colormap for 4 states
    cmap = ListedColormap(['lightgrey', 'green', 'orange', 'black'])
    bounds = [0,1,2,3,4]

    for idx in range(nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        ax = axs[r, c]
        ax.axis('off')
        if idx < S:
            arr = arrays[idx]
            im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=3, origin='upper')
            ax.set_title(f'Step {idx+1}', fontsize=12)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f'Saved figure to {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'ToyModelData.txt'
    # if a relative path isn't found, try DataAndAnalysis location
    import os
    if not os.path.isfile(path):
        alt = os.path.join(os.path.dirname(__file__), 'ToyModelData.txt')
        if os.path.isfile(alt):
            path = alt

    arrays = parse_toymodel_file(path)
    # default output file next to data file
    out_png = os.path.splitext(path)[0] + '_steps.png'
    plot_steps(arrays, out_png)
