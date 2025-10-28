import os
import sys
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from omegaconf import OmegaConf

# ensure repo root is importable when running the script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.vindr_dataset import VinDrDataset
from model.model import get_model


def make_transforms(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / max(1, total_examples)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

    acc = correct / total_examples if total_examples else 0.0
    avg_loss = total_loss / max(1, total_examples)
    return avg_loss, acc


def save_checkpoint(state: dict, save_dir: str, epoch: int):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth')
    torch.save(state, path)
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images-dir', required=True)
    p.add_argument('--annotations-file', required=True)
    p.add_argument('--view', default='CC')
    p.add_argument('--train-split', default='training')
    p.add_argument('--val-split', default='test')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--model-name', default='resnet18')
    p.add_argument('--num-classes', type=int, default=6)
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--in-channels', type=int, default=1)
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save-dir', default='checkpoints')
    p.add_argument('--workers', type=int, default=4)
    # These are not intended to be set via CLI but are passed by hydra wrapper
    p.add_argument('--optimizer-cfg', type=lambda x: None)
    p.add_argument('--scheduler-cfg', type=lambda x: None)
    # Weights & Biases
    p.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging')
    p.add_argument('--wandb-project', type=str, default=None, help='W&B project name')
    p.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (team/user)')
    p.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    # Label mapping options
    p.add_argument('--label-map-file', type=str, default=None, help='Path to JSON file containing label->int mapping')
    p.add_argument('--label-values', type=str, default=None, help='Comma-separated ordered list of label values to map to 0..N-1')
    p.add_argument('--auto-labels', type=str, choices=['auto', 'birads', 'none'], default='auto', help="How to build label mapping: 'auto' detects unique labels from annotations, 'birads' uses common BI-RADS ordering, 'none' attempts integer coercion and errors if it fails")
    p.add_argument('--use-class-weights', action='store_true',
                        help="Use inverse frequency class weighting for the loss function.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    transform = make_transforms(args.image_size)

    # Initialize W&B if requested
    wandb = None
    if args.use_wandb:
        try:
            import wandb as _wandb
        except Exception as e:
            print('Error: --use-wandb was passed but wandb import failed:', e)
            print('Install wandb (`pip install wandb`) or run without --use-wandb')
            sys.exit(1)
        # initialize
        wandb = _wandb
        wb_project = args.wandb_project or 'mammograph'
        wandb.init(project=wb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

    # Datasets
    train_ds = VinDrDataset(args.images_dir, args.annotations_file, split=args.train_split, view=args.view, transform=transform, )
    val_ds = VinDrDataset(args.images_dir, args.annotations_file, split=args.val_split, view=args.view, transform=transform)

    # Build label mapping according to CLI options
    label_map = None
    if args.label_map_file:
        import json
        with open(args.label_map_file, 'r') as f:
            label_map = json.load(f)
    elif args.label_values:
        vals = [v.strip() for v in args.label_values.split(',') if v.strip()]
        label_map = {v: i for i, v in enumerate(vals)}
    elif args.auto_labels == 'birads':
        # common BI-RADS categories ordering (user may need to adapt names)
        birads = ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5', 'BI-RADS 6']
        label_map = {v: i for i, v in enumerate(birads)}
    elif args.auto_labels == 'auto':
        # detect unique labels from the training annotations CSV limited to the requested split
        try:
            df = pd.read_csv(args.annotations_file)
            # filter by split and view if columns exist
            if 'split' in df.columns:
                df = df[df['split'] == args.train_split]
            if 'view' in df.columns:
                df = df[df['view'] == args.view]
            if 'breast_birads' in df.columns:
                unique = pd.Series(df['breast_birads'].dropna().unique()).astype(str).tolist()
            elif 'annotation' in df.columns:
                unique = pd.Series(df['annotation'].dropna().unique()).astype(str).tolist()
            else:
                unique = []
            unique = sorted(unique)
            if unique:
                label_map = {v: i for i, v in enumerate(unique)}
        except Exception:
            label_map = None

    # if label_map is still None and auto_labels == 'none' or detection failed, we'll let LabelWrappedDataset try coercion
    train_ds.label_map = label_map
    val_ds.label_map = label_map

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
    model = get_model(model_name=args.model_name, num_classes=args.num_classes, pretrained=args.pretrained, in_channels=args.in_channels, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    # Optimizer and Scheduler
    optim_cfg = getattr(args, 'optimizer_cfg', None)
    sched_cfg = getattr(args, 'scheduler_cfg', None)

    if optim_cfg and hasattr(torch.optim, optim_cfg.name):
        optimizer = getattr(torch.optim, optim_cfg.name)(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optim_cfg.lr
        )
    else:
        # fallback to default
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    scheduler = None
    if sched_cfg and hasattr(torch.optim.lr_scheduler, sched_cfg.name):
        raw_params = OmegaConf.to_container(sched_cfg.params, resolve=True) if sched_cfg.params else {}
        if isinstance(raw_params, dict):
            scheduler_params = {str(k): v for k, v in raw_params.items()}
        else:
            scheduler_params = {}
        scheduler = getattr(torch.optim.lr_scheduler, sched_cfg.name)(optimizer, **scheduler_params)

    class_weights: Optional[torch.Tensor] = None
    if args.use_class_weights:
        # compute class weights based on training set distribution
        class_distribution = train_ds.get_class_distribution()
        if class_distribution.numel() != args.num_classes:
            print(f"Warning: detected class distribution size {class_distribution.numel()} does not match num_classes={args.num_classes}; skipping class weights")
        else:
            total = class_distribution.sum().item()
            weights = [total / (count.item() + 1e-6) for count in class_distribution]
            class_weights = torch.tensor(weights, dtype=torch.float).to(device)
            print("Using class weights:", class_weights.tolist())
            
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}')
        # log to wandb if enabled
        if wandb is not None:
            log_data = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc}
            if scheduler:
                log_data['lr'] = scheduler.get_last_lr()[0]
            wandb.log(log_data)
        # save checkpoint
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args),
        }
        path = save_checkpoint(state, args.save_dir, epoch)
        print('Saved checkpoint:', path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.save_dir, 'best.pth')
            torch.save(state, best_path)
            print('Saved best checkpoint:', best_path)
            if wandb is not None:
                try:
                    # upload best model file to W&B run
                    wandb.save(best_path)
                    # also record summary
                    if hasattr(wandb, 'run') and wandb.run is not None:
                        wandb.run.summary['best_val_loss'] = best_val
                except Exception as e:
                    print('Warning: failed to save artifact to W&B:', e)


if __name__ == '__main__':
    main()
