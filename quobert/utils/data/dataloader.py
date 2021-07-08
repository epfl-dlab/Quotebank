import json
from functools import partial
from typing import Dict, List

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def __collate_batch(
    data: List[pd.Series], *, train: bool, test: bool
) -> Dict[str, torch.Tensor]:
    """
    Transform a list of `ConcatParquetDataset` entries into a Dict of tensors that can be fed to a BERT model.

    This private method is intented to be used with a partial to set `train` and then be fed to `torch.utils.data.DataLoader` as `collate_fn`
    
    Args:
        data (List[pd.Series]): a list of `ConcatParquetDataset` or `ParquetDataset` entries
        train (bool): set to `True` if `start_offset` and `end_offset` will be returned
    
    Returns:
        Dict[str, torch.Tensor]: a dict containing the dataset / sample index, input_ids, mask and if `train` the start and end offset
    """
    input_ids = pad_sequence(
        [torch.tensor(d.input_ids, dtype=torch.long) for d in data], batch_first=True,
    )  # (b_size, max_sentence_len)
    attention_mask = input_ids.where(
        input_ids == 0, torch.tensor(1)
    )  # (b_size, max_sentence_len)

    mask_ids = pad_sequence(
        [torch.tensor(d.mask_idx, dtype=torch.long) for d in data],
        batch_first=True,
        padding_value=-1,
    )  # mask for indexes of CLS/MASK tokens

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_ids": mask_ids,
    }

    if train:
        targets = torch.tensor([d.target for d in data], dtype=torch.long)  # (b_size, )
        out["targets"] = targets

    if test:
        entities = [json.loads(d.entities) for d in data]
        speakers = [d.speaker if "speaker" in d else "" for d in data]
        uid = [d.uid for d in data]
        out.update({"entities": entities, "speakers": speakers, "uid": uid})

    return out


collate_batch_train = partial(__collate_batch, train=True, test=False)
collate_batch_eval = partial(__collate_batch, train=False, test=True)
