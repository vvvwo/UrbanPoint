import os
import time

import numpy as np
import torch
import torch.utils.data
from functools import partial
from utils.logger import get_logger, close_logger
from utils.model import ModelBuilder
from utils.datasets import get_dataset, collate_fn
from utils.losses import LossBuilder
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
from os.path import join
from utils.helper_ply import write_ply, read_ply
from utils.misc import offset2batch
from shutil import move


class Tester:
    def __init__(self, cfg):
        self.epoch = 0
        self.start_epoch = 0
        self.best_iou = 0
        self.worst_iou = np.inf
        self.best_epoch = 0
        self.worst_epoch = 0
        self.sum_iou = 0
        self.max_epoch = cfg.epoch
        t = time.localtime()
        self.logger = get_logger(join(cfg.save_path, f"test_{t.tm_year}_{t.tm_mon}_{t.tm_mday}.log"), )
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.model = self.build_model()
        self.test_loader = self.build_test_loader()
        self.loss = self.build_loss()

    def test(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.model.eval()
            self.logger.info(f"################## Start Test ##################")
            for step, input_dict in enumerate(self.test_loader):
                print(input_dict['name'][0])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    output_dict = self.model(input_dict)
                output = output_dict["seg_logits"]
                pred = output.max(1)[1]
                self.save_pc(input_dict, pred)
                info = "Evaluating: [{}/{}]".format(step + 1, len(self.test_loader))
                print(info)
        self.logger.info("################## Finished ##################")

    def save_pc(self, input_dict, pred):
        vis_path = join(self.cfg.save_path, 'save')
        os.makedirs(vis_path) if not os.path.exists(vis_path) else None
        inverse = input_dict['inverse']
        origin_coord = input_dict['origin_coord']
        origin_color = input_dict['origin_color']

        name = input_dict['name'][0]
        save_path = join(vis_path, name)
        preds = pred[inverse].cpu().numpy().astype(np.uint8)
        # preds = pred.cpu().numpy().astype(np.uint8)
        coord = origin_coord.cpu().numpy()
        color = origin_color.cpu().numpy().astype(np.uint8)
        write_ply(save_path, [coord, color, preds],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'pred'])

    def build_model(self):
        model = ModelBuilder(self.cfg.model).cuda()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    key = key[7:]
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        batch_size = self.cfg.batch_size
        # batch_size = self.cfg.batch_size
        test_data = get_dataset(self.cfg.data["test"], batch_size, self.cfg.data_path_list)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )
        return test_loader

    def build_loss(self):
        return LossBuilder(self.cfg.loss)
