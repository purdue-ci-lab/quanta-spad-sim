# Quanta SPAD simulator

Temporal upsampling of intensity videos and simulating single-photon quanta data from them.

## Setup

### Environment

[uv](https://docs.astral.sh/uv/) is used as a package manager.

```shell
uv sync
```

### File and directory setup

[RIFE model download](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing). This should download the folder `train_log` which contains the model python files and the model itself `flownet.pkl`.

Place this folder in `ECCV2022-RIFE/train_log`.

The data directory defined in `DEFAULT_BASE_DIR` should follow the following format:

```
data_base_dir/
├─ separated_videos/
│  ├─ videos_manifest.yaml
│  ├─ video_subfolder/
├─ upsampled/
├─ upsampling_vid_idxs.yaml
```

SPAD data zarrs should save to `upsampled/`.

`videos_manifest.yaml` should follow the following format:

```yaml
- video_idx: 0
  subfolder: video_subfolder
  srcfps: 1000
  rife_interp_exp: 4
  linear_interp: 8
  linearize: true
- video_idx: 1
...
```

`upsampling_vid_idxs.yaml` is for having multiple videos upsampled iteratively.

```yaml
idxs:
  - 0
  - 1
  - 2
...
```

## Running

```shell
python generate_spad_dataset.py --video-idx <idx>
# or
uv run generate_spad_dataset.py --video-idx <idx>
```

If you want to convert an mp4 to a folder of images, define the file paths and spatial downsampling within `mp4_to_frames.py` before running

```shell
python mp4_to_frames.py
# or
uv run mp4_to_frames.py
```
