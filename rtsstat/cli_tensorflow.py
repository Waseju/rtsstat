import os
import sys

import click
import numpy as np
import tensorflow as tf
from rich import print, traceback

WD = os.path.dirname(__file__)


@click.command()
@click.option('-i', '--input', required=True, type=str, help='Path to data file to predict.')
@click.option('-m', '--model', type=str, help='Path to an already trained XGBoost model. If not passed a default model will be loaded.')
@click.option('-c/-nc', '--cuda/--no-cuda', type=bool, default=False, help='Whether to enable cuda or not')
@click.option('-o', '--output', type=str, help='Path to write the output to')
def main(input: str, model: str, cuda: bool, output: str):
    """Command-line interface for rtsstat"""

    print(r"""[bold blue]
        rtsstat
        """)

    print('[bold blue]Run [green]rtsstat --help [blue]for an overview of all commands\n')
    if not model:
        model = get_tensorflow_model(f'{WD}/models/tensorflow_test_model')
    else:
        model = get_tensorflow_model(model)
    if cuda:
        model.cuda()
    print('[bold blue] Parsing data')
    data_to_predict = read_data_to_predict(input)
    print('[bold blue] Performing predictions')
    predictions = np.round(model.predict(data_to_predict))
    print(predictions)
    if output:
        print(f'[bold blue]Writing predictions to {output}')
        write_results(predictions, output)


def read_data_to_predict(path_to_data_to_predict: str):
    """
    Parses the data to predict and returns a full Dataset include the DMatrix
    :param path_to_data_to_predict: Path to the data on which predictions should be performed on
    """
    return


def write_results(predictions: np.ndarray, path_to_write_to) -> None:
    """
    Writes the predictions into a human readable file.
    :param predictions: Predictions as a numpy array
    :param path_to_write_to: Path to write the predictions to
    """
    pass


def get_tensorflow_model(path_to_tensorflow_model: str):
    """
    Fetches the model of choice and creates a booster from it.
    :param path_to_tensorflow_model: Path to the xgboost model1
    """
    strategy = tf.distribute.MirroredStrategy()
    # Loading the model using lower level API
    with strategy.scope():
        model = tf.saved_model.load(path_to_tensorflow_model)

        return model


if __name__ == "__main__":
    traceback.install()
    sys.exit(main())  # pragma: no cover
