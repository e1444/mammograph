"""Hydra entrypoint for training. This wraps the existing argparse-based train script.

Run with: python -m src.train_hydra
Requires hydra-core: pip install hydra-core
"""
import os
import sys
from argparse import Namespace

# Ensure the `src` directory is on sys.path so we can import `src.train`
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from omegaconf import DictConfig, OmegaConf
import hydra

import src.train as train_module


def dictconfig_to_namespace(cfg: DictConfig) -> Namespace:
    # Flatten the nested structure into a flat namespace similar to argparse
    d = {}
    # dataset
    d['images_dir'] = cfg.dataset.images_dir
    d['annotations_file'] = cfg.dataset.annotations_file
    d['view'] = cfg.dataset.view
    d['train_split'] = cfg.dataset.train_split
    d['val_split'] = cfg.dataset.val_split
    # model
    d['model_name'] = cfg.model.name
    d['num_classes'] = cfg.model.num_classes
    d['pretrained'] = cfg.model.pretrained
    d['in_channels'] = cfg.model.in_channels
    d['freeze_backbone'] = cfg.model.freeze_backbone
    # training
    d['image_size'] = cfg.training.image_size
    d['batch_size'] = cfg.training.batch_size
    d['epochs'] = cfg.training.epochs
    d['lr'] = cfg.optimizer.lr
    d['device'] = cfg.training.device if cfg.training.device != 'auto' else ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    d['save_dir'] = cfg.training.save_dir
    d['workers'] = cfg.training.workers
    # wandb
    d['use_wandb'] = cfg.wandb.use
    d['wandb_project'] = cfg.wandb.project
    d['wandb_entity'] = cfg.wandb.entity
    d['wandb_run_name'] = cfg.wandb.run_name

    # pass through optimizer and scheduler configs
    d['optimizer_cfg'] = cfg.optimizer
    d['scheduler_cfg'] = cfg.scheduler

    # label mapping options (allow list or comma-string for label_values)
    d['label_map_file'] = cfg.dataset.label_map_file
    lv = cfg.dataset.get('label_values', None)
    if isinstance(lv, (list, tuple)):
        lv = ','.join(map(str, lv))
    d['label_values'] = lv
    d['auto_labels'] = cfg.dataset.auto_labels
    d['use_class_weights'] = cfg.dataset.use_class_weights

    return Namespace(**d)


@hydra.main(config_path="../conf", config_name="config")
def hydra_entry(cfg: DictConfig):
    print('Hydra config:\n', OmegaConf.to_yaml(cfg))
    args = dictconfig_to_namespace(cfg)
    # Monkeypatch the parse_args function in src.train to return our Namespace
    train_module.parse_args = lambda: args
    # Call the existing main() implementation from src.train
    train_module.main()


if __name__ == '__main__':
    hydra_entry()
