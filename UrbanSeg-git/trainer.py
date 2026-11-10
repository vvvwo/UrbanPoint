import os
import time

import numpy as np
import torch
import torch.utils.data
# from torch.profiler import profile, record_function, ProfilerActivity

from collections import OrderedDict
from functools import partial
from utils.logger import get_logger
from utils.model import ModelBuilder
from utils.datasets import get_dataset, point_collate_fn, collate_fn
from utils.losses import LossBuilder
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(self, cfg):
        self.epoch = 0
        self.start_epoch = 0
        self.best_iou = 0
        self.best_epoch = 0
        self.is_best = False
        self.max_epoch = cfg.epoch
        t = time.localtime()
        self.logger = get_logger(os.path.join(cfg.save_path, f"train_{t.tm_year}_{t.tm_mon}_{t.tm_mday}.log"), )
        self.logger.info("=> Loading config ...")
        self.logger.info(f"Random seed:{cfg.seed}")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building misc ...")
        self.loss = self.build_loss()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.check_resume()

    def train(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()
            self.logger.info(f"################## Epoch {self.epoch} ##################")
            for (steps, input_dict) in enumerate(self.train_loader):
                t_start = time.time()
                max_steps = len(self.train_loader)
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
                    output_dict = self.model(input_dict)
                    loss = self.loss(output_dict['seg_logits'], input_dict["segment"])
                # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
                self.optimizer.zero_grad()
                if self.cfg.enable_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)

                    # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                    # Fix torch warning scheduler step before optimizer step.
                    scaler = self.scaler.get_scale()
                    self.scaler.update()
                    if scaler <= self.scaler.get_scale():
                        self.scheduler.step()
                else:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                # log
                t_end = time.time()
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                info = "Training: [{}/{}] Loss: {:.4f} lr: {:.5f} --- {:.2f}ms/step".format(steps + 1, max_steps, loss,
                                                                                            lr,
                                                                                            1000 * (t_end - t_start))
                print(info)
            if self.epoch % 1 == 0:
                self.eval()
            self.save_checkpoint()
        self.logger.info(">>>>>>>>>>>>>>>> Finished >>>>>>>>>>>>>>>>")

    def eval(self):
        self.model.eval()
        num_classes = self.cfg.data['num_classes']
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        t_start = time.time()
        for step, input_dict in enumerate(self.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["seg_logits"]
            loss = self.loss(output_dict['seg_logits'], input_dict["segment"])
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            conf_matrix += confusion_matrix(pred.cpu(), segment.cpu(), labels=np.arange(0, num_classes))
            info = "Evaluating: [{}/{}] Loss: {:.4f} ".format(step + 1, len(self.val_loader), loss.item())
            print(info)
        t_end = time.time()
        gt_classes = np.sum(conf_matrix, axis=1)
        positive_classes = np.sum(conf_matrix, axis=0)
        true_positive_classes = np.diagonal(conf_matrix)
        iou_class = true_positive_classes / (gt_classes + positive_classes - true_positive_classes + 1e-10)
        acc_class = true_positive_classes / (gt_classes + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        oa = true_positive_classes.sum() / (conf_matrix.sum() + 1e-10)
        self.logger.info(
            "Val result: mIoU/mAcc/oa {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, oa
            )
        )
        for step in range(self.cfg.data['num_classes']):
            self.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=step,
                    name=self.cfg.data['names'][step],
                    iou=iou_class[step],
                    accuracy=acc_class[step],
                )
            )
        if self.best_iou < m_iou:
            self.is_best = True
            self.best_epoch = self.epoch
        self.best_iou = max(self.best_iou, m_iou)
        self.logger.info("Mean val step time: {:.2f}ms/step".format(1000 * (t_end - t_start) / len(self.val_loader)))
        self.logger.info("Best mIoU: {:.4f} in epoch {}".format(self.best_iou, self.best_epoch))

    def save_checkpoint(self):
        last_filename = os.path.join(self.cfg.save_path, "model", "model_last.pth")
        best_filename = os.path.join(self.cfg.save_path, "model", "model_best.pth")
        torch.save(
            {
                "epoch": self.epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": (
                    self.scaler.state_dict()
                    if self.cfg.enable_amp
                    else None
                ),
            },
            last_filename,
        )
        if self.is_best:
            self.is_best = False
            torch.save(
                {
                    "epoch": self.epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "scaler": (
                        self.scaler.state_dict()
                        if self.cfg.enable_amp
                        else None
                    ),
                },
                best_filename,
            )

    def build_model(self):
        model = ModelBuilder(self.cfg.model).cuda()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        return model

    def build_train_loader(self):
        train_data = get_dataset(self.cfg.data["train"], self.cfg.batch_size)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            sampler=None,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
        )
        return train_loader

    def build_val_loader(self):
        val_data = get_dataset(self.cfg.data["val"], self.cfg.batch_size)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            sampler=None,
        )
        return val_loader

    def build_optimizer(self):
        optimizer_cfg = self.cfg.optimizer.copy()
        optimizer_type = optimizer_cfg.pop('type')

        param_dicts = self.cfg.param_dicts
        params = [dict(names=[], params=[])]
        params += [dict(names=[], params=[], lr=param_dicts[i]['lr']) for i in range(len(param_dicts))]
        for n, p in self.model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i]["keyword"] in n:
                    params[i + 1]["names"].append(n)
                    params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                params[0]["names"].append(n)
                params[0]["params"].append(p)

        return getattr(torch.optim, optimizer_type)(params, **optimizer_cfg)

    def build_scheduler(self):
        self.cfg.scheduler["total_steps"] = len(self.train_loader) * self.cfg.epoch
        scheduler_cfg = self.cfg.scheduler.copy()
        scheduler_type = scheduler_cfg.pop('type')
        return getattr(torch.optim.lr_scheduler, scheduler_type)(self.optimizer, **scheduler_cfg)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler

    def build_loss(self):
        return LossBuilder(self.cfg.loss)

    def check_resume(self):
        if self.cfg.resume:
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(
                self.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    key = key[7:]
                weight[key] = value
            self.model.load_state_dict(weight, strict=True)
            self.logger.info(
                f"Resuming train at eval epoch: {checkpoint['epoch']}"
            )
            self.start_epoch = checkpoint["epoch"]
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
