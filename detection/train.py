# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
from torch import nn

import mmcv
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import torch
from mmcv import Config, DictAction
import torch.distributed as dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from the latest checkpoint automatically",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file.",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))

    torch.cuda.set_device(int(os.environ.get('SLURM_LOCALID')))
    os.environ["LOCAL_RANK"] = os.environ.get('SLURM_LOCALID')
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    cfg.device = 'cuda'
    cfg.gpu_ids = range(world_size)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume

    seed = init_random_seed(args.seed)
    set_random_seed(seed + rank, deterministic=args.deterministic)
    cfg.seed = seed

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
    )
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES
        )

    # for name, submodule in model.backbone.encoder.named_children():
    #     if "blocks" in name and isinstance(submodule, nn.Sequential):
    #         for i, block in enumerate(submodule):
    #             compiled_block = torch.compile(block)
    #             submodule[i] = compiled_block
    #         setattr(model.backbone.encoder, name, submodule)
    #     else:
    #         compiled_submodule = torch.compile(submodule)
    #         setattr(model.backbone.encoder, name, compiled_submodule)

    train_detector(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta={"env_info": collect_env(), "config": cfg.pretty_text, "seed": seed},
    )


if __name__ == "__main__":
    main()