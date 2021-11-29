import itertools
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tiff
from rich import print, traceback
from statannot import add_stat_annotation


@click.command()
@click.option('-m', '--meta', required=True, type=str, help='Path to metadata file')
@click.option('-r', '--ratios', required=True, type=str, help='Path to metadata file')
@click.option('-s', '--segs', required=True, type=str, help='Path to metadata file')
@click.option('-o', '--output', type=str, help='Path to write the output to')
def main(meta: str, ratios: str, segs: str, output: str):
    """Command-line interface for rtsstat"""

    print(r"""[bold blue]
        rtsstat
        """)

    print('[bold blue]Run [green]rtsstat --help [blue]for an overview of all commands\n')
    df = pd.read_csv(meta, header=0)
    df = df
    ratios = [calc_ratio(ratios, segs, x) for x in df['Filename']]
    df["Ratio"] = ratios
    df["Breeding Line and Treatment"] = df["Breeding Line"] + " " + df["Treatment"]
    plt.figure()
    ax = sns.boxplot(x="Breeding Line and Treatment", y="Ratio",
                     data=df[df["Ratio"] != 0], showmeans=True)
    df = df.dropna()
    print(df.groupby(['Treatment', 'Breeding Line']).mean())
    print(df.groupby(['Treatment', 'Breeding Line']).std())
    box_pairs = []
    for bl in df["Breeding Line"].unique():
        bps = list(filter(lambda x: x.startswith(bl), df["Breeding Line and Treatment"].unique()))
        box_pairs = box_pairs + list(itertools.combinations(bps, 2))
    add_stat_annotation(ax, x="Breeding Line and Treatment", y="Ratio", data=df,
                        box_pairs=box_pairs,
                        test='t-test_welch', text_format='star', loc='inside', verbose=2)
    plt.tight_layout()
    plt.savefig('./boxplot.pdf'.replace(" ", ""), bbox_inches='tight')
    plt.close()


def calc_ratio(ratios, segs, x):
    ratio_img = tiff.imread(ratios + x + "_ratio.tif")
    ratio_img = np.nan_to_num(ratio_img)
    tif_img = np.load(segs + x + ".npy")
    ratio = extract_ph(ratio_img, tif_img, 2)
    if np.isnan(ratio):
        return np.nan
    else:
        return ratio


def evaluate_img(img_path, class_index, seg_path):
    ratio_img = tiff.imread(img_path)
    print(ratio_img.shape)
    seg_img = np.load(seg_path)
    ratio = extract_ph(ratio_img, seg_img, class_index)
    if np.isnan(ratio):
        return []
    else:
        return ratio


def extract_ph(ratio_img, tif_img, class_index):
    late_array = ratio_img[tif_img == class_index]
    # Empirically determined value to exclude too small predictions.
    if late_array.shape[0] < 11000:
        return np.nan
    return np.true_divide(late_array.sum(0), (late_array != 0).sum(0))


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
