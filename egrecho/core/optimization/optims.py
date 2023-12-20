# -*- coding:utf-8 -*-
# Copyright xmuspeech (Author: Leo 2023-09)

from functools import partial

from torch import optim

from egrecho.utils.register import Register

TORCH_OPTIMIZERS = Register("optimizer")

optims = {
    k.lower(): v
    for k, v in optim.__dict__.items()
    if not k.startswith("__") and k[0].isupper() and k != "Optimizer"
}
for name, opt_cls in optims.items():
    if name == "sgd":

        def wrapper(opt_cls, parameters, lr=None, **kwargs):
            if lr is None:
                raise TypeError(
                    "The `learning_rate` argument is required when the optimizer is SGD."
                )
            return opt_cls(parameters, lr, **kwargs)

        opt_cls = partial(wrapper, opt_cls)
    TORCH_OPTIMIZERS.register(opt_cls, name=name)
