"""WikiQA data loader."""

import typing
import csv
from pathlib import Path

import pandas as pd

import matchzoo
from matchzoo.engine.base_task import BaseTask

_url = "https://download.microsoft.com/download/msmarco" 


def load_data(
    stage: str = 'train',
    task: typing.Union[str, BaseTask] = 'ranking',
    filtered: bool = False,
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load msmarco data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    #data_root = _download_data()
    data_root = Path(matchzoo.USER_DATA_DIR)
    file_path = data_root.joinpath(f'msmarco/msmarco-{stage}.tsv')
    print(file_path)
    data_pack = _read_data(file_path, task)

    if task == 'ranking' or isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif task == 'classification' or isinstance(
            task, matchzoo.tasks.Classification):
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data():
    ref_path = matchzoo.utils.get_file(
        'msmarco', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='msmarco'
    )
    return Path(ref_path).parent.joinpath('zipdata')


def _read_data(path, task):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    df = pd.DataFrame({
        'text_left': table['querystring'],
        'text_right': table['documentstring'],
        'id_left': table['topicid'],
        'id_right': table['docid'],
        'label': table['label']
    })
    return matchzoo.pack(df, task)

if __name__ == '__main__':
    print(load_data())
    
