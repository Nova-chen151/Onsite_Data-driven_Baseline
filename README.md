# OnSite 2025 PKL inference

[中文说明](README_zh.md)

Generate background-vehicle trajectories from OnSite 2025 `*_exam.pkl` and OpenDRIVE (`.xodr`) scenarios with the adapted SMART model. This branch outputs `*_output.pkl` and can render GIFs.

## Input and checkpoint

Place the test data and model checkpoint as follows:

```text
data/B/<scene>/<scene>_exam.pkl
data/B/<scene>/<scene>.xodr
ckpt/epoch=07-step=30440-val_loss=2.52.ckpt
```

Obtain the [checkpoint](https://yunpan.tongji.edu.cn/link/AA37DAF55B73094639BF22C8B9FCD0CD38) separately. Data, checkpoint and generated results are not included in this repository. The model's two token files are included in `smart/tokens/` and are required for inference.

Empty `.gitkeep` files retain these directories in Git; put scenario folders under `data/B/` and the checkpoint under `ckpt/`. Results are written to `results/pkl_2025/`.

## Environment

The supplied `requirements.txt` records the original model environment. Inference has been tested locally in the existing `catk` conda environment with Python 3.11 and PyTorch 2.4.1. PyTorch Geometric, `torch-scatter` and `torch-cluster` require builds compatible with the installed PyTorch and CUDA versions; see the comments in `requirements.txt`. The current inference implementation runs on CPU.

## Run

```bash
python inference_pkl_2025.py --test_dir data/B --output_dir results/pkl_2025 --limit 10
python visualize_pkl_gif_2025.py --data_dir data/B --output_dir results/pkl_2025
```

Omit `--limit 10` to process every scenario. Add `--skip_existing` to resume based on existing output filenames; existing files are not revalidated. Use `--ckpt_path` to select another checkpoint. The GIFs are saved in `results/pkl_2025/gifs/`.

The input is expected to use 10 Hz, with 31 historical frames and arrays sized by `sim_duration`. The script preserves history and non-predicted agents, and fills the future of agents selected by `predict_mask`. In these test scenes Ego has no valid future in the input and disappears after the history in the GIF. The underlying model still rolls out all agents internally, including Ego. GIFs show the output validity mask; they are qualitative visualizations, not benchmark scores.
