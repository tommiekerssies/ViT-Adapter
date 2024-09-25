install CUDA and set CUDA_HOME environment variable
conda create -n openmmlab python  
conda activate openmmlab  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 (change to right cuda version)  
pip install mmcv-full scipy timm mmdet==2.25.0  
pip install git+https://github.com/cocodataset/panopticapi.git  
cd detection/ops  
sh make.sh  

put coco in detection/data in this structure:
```none
detection
├── data
│   ├── coco
│   │   ├── annotations
|   |   |   ├── instances_train2017.json
|   |   |   ├── instances_val2017.json
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

modify /home/tkerssies/miniconda3/envs/openmmlab/lib/python3.12/site-packages/mmdet/__init__.py (change to right path)
```
# assert (mmcv_version >= digit_version(mmcv_minimum_version)
#         and mmcv_version <= digit_version(mmcv_maximum_version)), \
#     f'MMCV=={mmcv.__version__} is used but incompatible. ' \
#     f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'
```

modify /home/tkerssies/miniconda3/envs/openmmlab/lib/python3.12/site-packages/torch/nn/parallel/_functions.py (change to right path)
```
def _get_stream(device: torch.device):
    """Get a background stream for copying between CPU and target device."""
    global _streams
    device_mod = getattr(torch, "cuda", None)
    if device_mod is None:
        return None
    if _streams is None:
        _streams = [None] * device_mod.device_count()
    if _streams[device] is None:
        _streams[device] = device_mod.Stream(device)
    return _streams[device]
```

modify /home/tkerssies/miniconda3/envs/openmmlab/lib/python3.12/site-packages/mmcv/parallel/distributed.py (change to right path)
```
module_to_run = self.module
```

use sbatch slurm_train.sh to train DINOv2reg4-L for panoptic segmentation on 16 h100 GPUs with 4 gpus per node on Snellius