class AttrDict(dict):
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:
        self[key] = value


config = AttrDict()

config.model_args = AttrDict()

config.dataset_args = AttrDict(
    features="M",
    target="OT",
    embed="timeF",
    percent=100,
    max_len=-1,
    freq="h",
    num_workers=0,
    label_len=0,
)

config.training_args = AttrDict(
    batch_size=256,
    decay_fac=0.75,
    ema=False,
    itr=1,
    learning_rate=1e-3,
    loss_func="mse",
    num_workers=0,
    patience=8,
    train_epochs=30,
    lr_schedule_config=AttrDict(
        type="type3",
        tmax=16,
    ),
)
