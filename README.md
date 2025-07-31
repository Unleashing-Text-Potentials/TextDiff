# TextDiff

## Requirements 
We use a Docker image for easier reproduction: `tiankaihang/azureml_docker:horovod`


## Getting Started

### General

1. Download pretrained models.

   CLIP-ViP-B/32: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D), Save to: `/blob_mount/path/to/CLIP-ViP-B/32/checkpoint`

   CLIP-ViP-B/16: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D), Save to: `/blob_mount/path/to/CLIP-ViP-B/16/checkpoint`

2. We trained models on the MSR-VTT, DiDeMo , Activitynet and LSMDC datasets. To download the datasets, refer to this [repository](https://github.com/ArrowLuo/CLIP4Clip), and configure the absolute dataset path in `launch_textdiff.sh`.

3. Download  generate feature [Download](https://drive.google.com/drive/folders/1vCfRlMCB5QgzliqC5eocUxj1BDJpkZdW?usp=sharing). Edit `launch_textdiff.sh` to replace the default `feature_path` value with your absolute download path.

4. Set up the environment for running the experiments.

Set `PATH_TO_STORAGE` to the absolute path of your `blob_mount` folder.
```bash
PATH_TO_STORAGE=/absolute/path/to/your/blob_mount
```
Clone this repo and launch the Docker container for running the experiments. 
If you want to pre-train on your own dataset, please prepare the environment with `horovod`. It is a better choice to use the pre-built docker image `tiankaihang/azureml_docker:horovod`. Or you can build from the [dockerfile](./docker/Dockerfile).
Using mixed-precision training hence GPUs with Tensor Cores are recommended.

```bash
source launch_textdiff.sh $PATH_TO_STORAGE
# update the transformers package
pip install --upgrade transformers
```

### Fine-tuning for text-video retrieval

```bash
# inside the container
horovodrun -np $NUM_GPUS python src/tasks/run_video_retrieval.py \
    --config $CONFIG_PATH 
```

`$CONFIG_PATH` should be set to one of the .json config files available at [src/configs](src/configs) postfixed with `_retrieval.json`. For example, you can use `src/configs/msrvtt_retrieval/msrvtt_retrieval_vip_base_32.json` on MSRVTT retrieval. For model, currently, `pretrain_vip_base_32.json` and `pretrain_vip_base_16.json` are supported. For dataset, MSR-VTT, DiDemo, LSMDC, ActivityNet Captions are supported.

## Acknowledgements
The code is based on [CLIP-VIP](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP).

