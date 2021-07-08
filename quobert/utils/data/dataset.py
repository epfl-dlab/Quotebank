import bisect
from typing import Tuple, List

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset


class ParquetDataset(Dataset):
    """
    Parquet Dataset is a wrapper around `torch.utils.data.Dataset`. It loads and serve a parquet DataFrame.
    
    Args:
        parquet_path (str): The path to a single parquet file, can be compressed. If using multiple file, check `ConcatParquetDataset`
        sample_n (int, optional): Set to the number of items from the data set to sample. If 0, use all items. Defaults to 0.
    """

    def __init__(
        self, parquet_path: str, sample_n: int = 0
    ):
        super(ParquetDataset, self).__init__()
        self.df = pd.read_parquet(parquet_path)
        if sample_n > 0:
            self.df = self.df.sample(n=sample_n)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]


class ConcatParquetDataset(ConcatDataset):
    """
    Concat Parquet Dataset is a wrapper around `torch.utils.data.ConcatDataset`. It serves multiple `ParquetDataset`
    
    Args:
        datasets (List[ParquetDataset]): a list of `ParquetDataset`
    """

    def __init__(self, datasets: List[ParquetDataset]):
        super(ConcatParquetDataset, self).__init__(datasets)

    def __getitem__(self, idx: int) -> Tuple[int, int, pd.Series]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
