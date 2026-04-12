from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from utils import get_regularization


class BaseModel(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--model_path", type=str, default="", help="Model save path.")
        parser.add_argument("--loss", type=str, default="", help="loss type")
        parser.add_argument("--l2_coef", type=float, default=0.0, help="coefficient of regularization term")
        return parser

    def __init__(self, args, reader, device):
        super().__init__()
        self.display_name = "BaseModel"
        self.reader = reader
        self.model_path = args.model_path
        self.loss_type = args.loss
        self.l2_coef = args.l2_coef
        self.no_reg = 0.0 < args.l2_coef < 1.0
        self.device = device
        self.args = args
        self._define_params(args, reader)
        self.sigmoid = nn.Sigmoid()

    def log(self):
        self.reader.log()
        print("Model params")
        print("\tmodel_path = " + str(self.model_path))
        print("\tloss_type = " + str(self.loss_type))
        print("\tl2_coef = " + str(self.l2_coef))
        print("\tdevice = " + str(self.device))

    def get_regularization(self, *modules):
        return get_regularization(*modules)

    def do_forward_and_loss(self, feed_dict):
        out_dict = self.forward(feed_dict)
        out_dict["loss"] = self.get_loss(feed_dict, out_dict)
        return out_dict

    def forward(self, feed_dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict["probs"] = nn.Sigmoid()(out_dict["preds"])
        return out_dict

    def wrap_batch(self, batch):
        for key, value in batch.items():
            if type(value).__module__ == np.__name__:
                batch[key] = torch.from_numpy(value)
            elif torch.is_tensor(value):
                batch[key] = value
            elif type(value) is list:
                batch[key] = torch.tensor(value)
            else:
                continue
            if batch[key].type() == "torch.DoubleTensor":
                batch[key] = batch[key].float()
            batch[key] = batch[key].to(self.device)
        return batch

    def save_checkpoint(self):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "args": self.args,
            },
            self.model_path + ".checkpoint",
        )
        print("Model (checkpoint) saved to " + self.model_path)

    def load_from_checkpoint(self, model_path, with_optimizer=True):
        print("Load (checkpoint) from " + model_path + ".checkpoint")
        try:
            checkpoint = torch.load(model_path + ".checkpoint", map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path + ".checkpoint", map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if with_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_path = model_path

    def _define_params(self, args, reader):
        raise NotImplementedError

    def get_forward(self, feed_dict):
        raise NotImplementedError

    def get_loss(self, feed_dict, out_dict):
        raise NotImplementedError
