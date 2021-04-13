import struct

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rsp.moco_r50.data import sentinel_augmentation_valid
import lz4


class StreamingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: int,
        inference_model: str = "moco",
        decompress_data: bool = False,
    ):
        """ load df of compressed np arrays representing imgs into memory"""
        self.df = df
        self.image_col = image_col
        self.inference_model = inference_model
        self.decompress_data = decompress_data

    def __len__(self):
        return len(self.df)

    def _load_patch_sentinel(self, img: np.ndarray):
        """ load and transform sentinel image patch to prep for model """
        img = img[:12].transpose(1, 2, 0) / 10_000
        return sentinel_augmentation_valid()(image=img)["image"]

    def _delete_array(self, idx):
        self.df.iat[idx, self.image_col] = None

    def __getitem__(self, idx):
        """ get next data point, decompress, transform for inference """
        img = self.df.loc[idx][self.image_col]

        if self.decompress_data:
            compressed_bytes = img.tobytes()
            decompressed_bytes = lz4.frame.decompress(compressed_bytes)
            storage_type, shape_0, shape_1, shape_2 = struct.unpack(
                "cIII", decompressed_bytes[:16]
            )
            img = np.frombuffer(decompressed_bytes[16:], dtype=storage_type)
            img = img.reshape(shape_0, shape_1, shape_2)

        if self.inference_model == "moco":
            img = self._load_patch_sentinel(img)
        else:
            img = torch.Tensor(img)

        self._delete_array(idx)
        return img
