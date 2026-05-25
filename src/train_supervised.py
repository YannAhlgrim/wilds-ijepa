import os
import shutil
import sys
import yaml
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.datasets.wilds import make_iwildcam
from src.helper import init_model
from src.models.head import ViTClassifier
from src.transforms import make_transforms, make_transform_eval
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter, resolve_log_dir
from src.utils.optimizers import LARS
from src.eval_wilds import main as eval_wilds_main

# --
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def distributed_average(value, device):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def distributed_sum(value, device):
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


class EarlyStopping:
    def __init__(
        self,
        enabled=True,
        patience=10,
        min_delta=0.0,
        min_epochs=0,
        restore_best_weights=True,
    ):
        self.enabled = enabled
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.restore_best_weights = restore_best_weights
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.best_state = None

    def state_dict(self):
        return {
            "enabled": self.enabled,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "restore_best_weights": self.restore_best_weights,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.enabled = state.get("enabled", self.enabled)
        self.patience = state.get("patience", self.patience)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.min_epochs = state.get("min_epochs", self.min_epochs)
        self.restore_best_weights = state.get(
            "restore_best_weights", self.restore_best_weights
        )
        self.best_metric = state.get("best_metric", self.best_metric)
        self.best_epoch = state.get("best_epoch", self.best_epoch)
        self.bad_epochs = state.get("bad_epochs", self.bad_epochs)

    def step(self, epoch, metric, model_module):
        if not self.enabled:
            return False, False

        improved = metric < (self.best_metric - self.min_delta)
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.bad_epochs = 0
            if self.restore_best_weights:
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model_module.state_dict().items()
                }
            return True, False

        self.bad_epochs += 1
        should_stop = (
            epoch + 1
        ) >= self.min_epochs and self.bad_epochs >= self.patience
        return False, should_stop

    def restore(self, model_module, device):
        if self.restore_best_weights and self.best_state is not None:
            model_module.load_state_dict(self.best_state)
            model_module.to(device)


def evaluate(model, loader, criterion, device, use_bfloat16):
    model.eval()
    loss_sum = 0.0
    n_correct = 0.0
    n_total = 0.0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_bfloat16, dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            preds = outputs.argmax(dim=1)
            n_correct += float((preds == labels).sum().item())
            n_total += float(batch_size)
            loss_sum += float(loss.item()) * float(batch_size)

    global_loss_sum = distributed_sum(loss_sum, device)
    global_correct = distributed_sum(n_correct, device)
    global_total = distributed_sum(n_total, device)

    val_loss = global_loss_sum / max(global_total, 1.0)
    val_acc = global_correct / max(global_total, 1.0)
    return val_loss, val_acc


def main(args, resume_preempt=False):
    del resume_preempt

    world_size, rank = init_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for supervised distributed training")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    m_args = args["meta"]
    o_args = args["optimization"]
    d_args = args["data"]
    mk_args = args["mask"]
    l_args = args["logging"]
    v_args = args["validation"]
    es_args = o_args["early_stopping"]

    folder = resolve_log_dir(args, stage="train")
    tag = l_args["write_tag"]

    with open(os.path.join(folder, "params-supervised.yaml"), "w") as f:
        yaml.dump(args, f)
    with open(os.path.join(folder, "params.yaml"), "w") as f:
        yaml.dump(args, f)

    save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
    latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
    best_path = os.path.join(folder, f"{tag}-best.pth.tar")
    log_file = os.path.join(folder, f"{tag}_r{rank}.csv")

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%.6f", "train_loss"),
        ("%.6f", "val_loss"),
        ("%.6f", "val_acc"),
        ("%.6e", "lr"),
        ("%.6f", "best_val_loss"),
        ("%d", "best_epoch"),
        ("%d", "early_stop"),
    )

    encoder, _ = init_model(
        device=device,
        patch_size=mk_args["patch_size"],
        crop_size=d_args["crop_size"],
        model_name=m_args["model_name"],
    )

    embed_dim = m_args["embed_dim"]
    model = ViTClassifier(
        encoder,
        m_args["num_classes"],
        embed_dim,
        probe_type=m_args.get("probe_type", "linear"),
        mlp_hidden_dim=m_args.get("mlp_hidden_dim"),
        dropout=m_args.get("dropout", 0.0),
    ).to(device)

    if o_args["freeze_weights"]:
        logger.info("Freezing encoder weights (Linear Probing mode)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder.eval()
    else:
        logger.info("Training full model (Fine-tuning mode)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = o_args["optimizer"].lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=o_args["lr"], weight_decay=o_args["weight_decay"]
        )
    elif optimizer_name == "lars":
        optimizer = LARS(
            params,
            lr=o_args["lr"],
            weight_decay=o_args["weight_decay"],
            momentum=o_args.get("momentum", 0.9),
            eta=o_args.get("lars_eta", 0.001),
            eps=o_args.get("lars_eps", 1e-8),
            exclude_bias_and_norm=o_args.get("lars_exclude_bias_and_norm", True),
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=o_args["lr"],
            momentum=o_args.get("momentum", 0.9),
            weight_decay=o_args["weight_decay"],
        )

    scheduler = None
    lr_schedule = o_args.get("lr_schedule", "cosine").lower()
    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=o_args.get("step_milestones", [15, 30, 45]),
            gamma=o_args.get("step_gamma", 0.1),
        )
    elif lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=o_args["epochs"], eta_min=o_args["final_lr"]
        )

    train_transform = make_transforms(
        crop_size=d_args["crop_size"],
        crop_scale=tuple(d_args["crop_scale"]),
        horizontal_flip=d_args["use_horizontal_flip"],
        color_distortion=d_args["use_color_distortion"],
        color_jitter=d_args["color_jitter_strength"],
        gaussian_blur=d_args["use_gaussian_blur"],
        use_random_resized_crop=d_args.get("use_random_resized_crop", True),
    )
    val_transform = make_transform_eval(
        crop_size=d_args["crop_size"],
    )

    _, train_loader, train_sampler = make_iwildcam(
        transform=train_transform,
        split="train",
        batch_size=d_args["batch_size"],
        root_path=d_args["root_path"],
        rank=rank,
        world_size=world_size,
        collator=None,
        num_workers=d_args["num_workers"],
        pin_mem=d_args["pin_mem"],
        drop_last=True,
    )

    _, val_loader, val_sampler = make_iwildcam(
        transform=val_transform,
        split="val",
        batch_size=d_args["batch_size"],
        root_path=d_args["root_path"],
        rank=rank,
        world_size=world_size,
        collator=None,
        num_workers=d_args["num_workers"],
        pin_mem=d_args["pin_mem"],
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss().to(device)
    model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

    early_stopper = EarlyStopping(
        enabled=es_args["enabled"],
        patience=es_args["patience"],
        min_delta=es_args["min_delta"],
        min_epochs=es_args["min_epochs"],
        restore_best_weights=es_args["restore_best_weights"],
    )

    start_epoch = 0

    checkpoint_to_load = None
    resuming_interrupted = False
    if os.path.exists(latest_path):
        checkpoint_to_load = latest_path
        resuming_interrupted = True
    elif m_args["load_checkpoint"]:
        r_file = m_args["read_checkpoint"]
        checkpoint_folder = m_args["checkpoint_folder"]
        if os.path.isabs(r_file):
            checkpoint_to_load = r_file
        else:
            checkpoint_to_load = os.path.join(checkpoint_folder, r_file)

    if checkpoint_to_load is not None and not os.path.exists(checkpoint_to_load):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_load}")

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        checkpoint = torch.load(checkpoint_to_load, map_location="cpu")
        if resuming_interrupted and "model" in checkpoint:
            model.module.load_state_dict(checkpoint["model"])
            if "opt" in checkpoint:
                optimizer.load_state_dict(checkpoint["opt"])
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint.get("epoch", 0))
            early_stopper.load_state_dict(checkpoint.get("early_stopping", {}))
            logger.info(
                f"Resuming training from {checkpoint_to_load} at epoch {start_epoch}"
            )
        else:
            encoder_state = checkpoint.get("encoder")
            if encoder_state is None and "model" in checkpoint:
                encoder_state = {
                    k.replace("encoder.", "", 1): v
                    for k, v in checkpoint["model"].items()
                    if k.startswith("encoder.")
                }
            if encoder_state is None:
                raise KeyError(
                    f"No encoder weights found in checkpoint: {checkpoint_to_load}"
                )
            encoder_state = strip_module_prefix(encoder_state)
            msg = model.module.encoder.load_state_dict(encoder_state, strict=False)
            logger.info(
                f"Loaded pre-trained encoder from {checkpoint_to_load} with msg: {msg}"
            )

    def save_checkpoint(epoch, train_loss, val_loss, val_acc, is_best=False):
        save_dict = {
            "model": model.module.state_dict(),
            "opt": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "args": args,
            "early_stopping": early_stopper.state_dict(),
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if epoch % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=epoch))
            if is_best:
                torch.save(save_dict, best_path)

    eval_every = int(v_args["eval_every"])

    for epoch in range(start_epoch, o_args["epochs"]):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model.train()
        if o_args["freeze_weights"]:
            model.module.encoder.eval()

        loss_meter = AverageMeter()

        for itr, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(
                enabled=m_args["use_bfloat16"], dtype=torch.bfloat16
            ):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), n=labels.size(0))

            if itr % log_freq == 0 and rank == 0:
                logger.info(
                    f"Epoch {epoch + 1} [{itr}/{len(train_loader)}] Train Loss: {loss_meter.avg:.4f}"
                )

        train_loss = distributed_average(loss_meter.avg, device)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == o_args["epochs"])
        if do_eval:
            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_bfloat16=m_args["use_bfloat16"],
            )
            is_best, should_stop = early_stopper.step(epoch + 1, val_loss, model.module)
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            is_best, should_stop = False, False

        if scheduler is not None:
            scheduler.step()

        if rank == 0:
            logger.info(
                f"Epoch {epoch + 1} done | train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f} best_val_loss={early_stopper.best_metric:.6f}"
            )

        csv_logger.log(
            epoch + 1,
            train_loss,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
            early_stopper.best_metric,
            early_stopper.best_epoch,
            int(should_stop),
        )

        save_checkpoint(epoch + 1, train_loss, val_loss, val_acc, is_best=is_best)

        stop_tensor = torch.tensor([int(should_stop)], device=device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(stop_tensor, src=0)

        if bool(stop_tensor.item()):
            if rank == 0:
                logger.info(
                    f"Early stopping at epoch {epoch + 1}. Best val_loss={early_stopper.best_metric:.6f} @ epoch {early_stopper.best_epoch}"
                )
            break

    if early_stopper.enabled and early_stopper.restore_best_weights:
        if rank == 0:
            logger.info("Restoring best model weights before exit")
        early_stopper.restore(model.module, device)
        if rank == 0:
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "epoch": early_stopper.best_epoch,
                    "val_loss": early_stopper.best_metric,
                    "args": args,
                },
                best_path,
            )

    if rank == 0:
        eval_args = {
            "meta": {
                "seed": m_args.get("seed", _GLOBAL_SEED),
                "model_name": m_args["model_name"],
                "embed_dim": m_args["embed_dim"],
                "num_classes": m_args["num_classes"],
                "patch_size": mk_args.get("patch_size", m_args.get("patch_size", 16)),
                "crop_size": d_args.get("crop_size", m_args.get("crop_size", 224)),
                "use_bfloat16": m_args.get("use_bfloat16", True),
                "checkpoint_path": best_path,
                "force_single_process": True,
            },
            "data": {
                "batch_size": d_args.get("batch_size", 128),
                "root_path": d_args.get("root_path", "./wilds_data"),
                "num_workers": d_args.get("num_workers", 8),
                "pin_mem": d_args.get("pin_mem", True),
                "split": "test",
                "download": True,
            },
            "logging": {
                "write_tag": "iwildcam_test",
                "auto_folder": True,
            },
        }
        eval_wilds_main(args=eval_args)

        eval_folder = eval_args.get("logging", {}).get("folder")
        supervised_params = os.path.join(folder, "params-supervised.yaml")
        if eval_folder and os.path.exists(supervised_params):
            try:
                shutil.copy2(
                    supervised_params,
                    os.path.join(eval_folder, "params-supervised.yaml"),
                )
                shutil.copy2(
                    supervised_params,
                    os.path.join(eval_folder, "params.yaml"),
                )
            except OSError:
                logger.warning("Could not copy supervised params to eval folder")

        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except OSError:
                logger.warning("Could not remove supervised run folder")


if __name__ == "__main__":
    raise RuntimeError(
        "Use main_distributed_supervised.py to launch this script with a config file."
    )
