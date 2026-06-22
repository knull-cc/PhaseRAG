from __future__ import annotations

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from PhaseRAG.utils.timefeatures import time_features


class DatasetETTHourMulti(Dataset):
    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: list[int] | None = None,
        features: str = "M",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        percent: int = 100,
        var_needed: int | None = None,
        **_kwargs,
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be one of train, val, test")

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.data_path = data_path
        self.features = features
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.scale = scale
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.target = target
        self.timeenc = timeenc
        self.var_needed = var_needed

        self.scaler = StandardScaler()
        self._read_data()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        start = index % self.tot_len
        seq_end = start + self.seq_len
        label_start = seq_end - self.label_len
        label_end = label_start + self.label_len + self.pred_len

        seq_x = self.data_x[start:seq_end, : self.var_needed]
        seq_y = self.data_y[label_start:label_end, : self.var_needed]
        seq_x_mark = self.data_stamp[start:seq_end]
        seq_y_mark = self.data_stamp[label_start:label_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def _read_data(self) -> None:
        data_file = os.path.join(self.root_path, self.data_path)
        raw = pd.read_csv(data_file)

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features in {"M", "MS"}:
            df_data = raw[raw.columns[1:]]
        elif self.features == "S":
            df_data = raw[[self.target]]
        else:
            raise ValueError("features must be one of M, MS, S")

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        stamp = raw[["date"]][border1:border2].copy()
        stamp["date"] = pd.to_datetime(stamp.date)
        if self.timeenc == 0:
            stamp["month"] = stamp["date"].dt.month
            stamp["day"] = stamp["date"].dt.day
            stamp["weekday"] = stamp["date"].dt.weekday
            stamp["hour"] = stamp["date"].dt.hour
            data_stamp = stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(
                pd.to_datetime(stamp["date"].values),
                freq=self.freq,
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
