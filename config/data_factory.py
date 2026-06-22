from __future__ import annotations

from torch.utils.data import DataLoader

from PhaseRAG.config.data_loader import DatasetETTHourMulti


DATASETS = {
    "ett_h": DatasetETTHourMulti,
}


def data_provider(args, flag: str, drop_last_test: bool = False):
    dataset_cls = DATASETS[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    scale = args.scale if "scale" in args else True
    noisy_ratio = 0.0 if flag == "test" else getattr(args, "noisy_ratio", 0.0)

    if flag == "train":
        shuffle = True
        drop_last = True
    else:
        shuffle = flag == "val"
        drop_last = drop_last_test

    dataset = dataset_cls(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        percent=args.percent,
        scale=scale,
        max_len=args.max_len,
        var_needed=args.var_needed,
        noisy_ratio=noisy_ratio,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return dataset, data_loader
