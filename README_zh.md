# OnSite 2025 PKL 推理

使用适配的 SMART 模型，根据 `*_exam.pkl` 和对应的 `.xodr` 路网生成背景车轨迹，输出 `*_output.pkl`，并支持 GIF 可视化。

## 数据与权重

```text
data/B/<场景名>/<场景名>_exam.pkl
data/B/<场景名>/<场景名>.xodr
ckpt/epoch=07-step=30440-val_loss=2.52.ckpt
```

[模型权重](https://yunpan.tongji.edu.cn/link/AA37DAF55B73094639BF22C8B9FCD0CD38)需单独下载；仓库不包含数据、权重和推理结果。`smart/tokens/` 中的两个令牌文件是推理必需文件。

仓库中的空 `.gitkeep` 文件用于保留输入目录：场景文件夹放在 `data/B/`，模型权重放在 `ckpt/`。输出写入 `results/pkl_2025/`。

## 运行

本机在已有 `catk` 环境中验证，Python 3.11、PyTorch 2.4.1；原模型环境依赖见 `requirements.txt`，其中 PyTorch Geometric、`torch-scatter` 和 `torch-cluster` 需要与 PyTorch / CUDA 版本匹配。当前实现使用 CPU。

```bash
python inference_pkl_2025.py --test_dir data/B --output_dir results/pkl_2025 --limit 10
python visualize_pkl_gif_2025.py --data_dir data/B --output_dir results/pkl_2025
```

去掉 `--limit 10` 可推理全部场景；`--skip_existing` 按文件名跳过已有输出，跳过时不重新验证；`--ckpt_path` 可指定权重。GIF 默认保存在 `results/pkl_2025/gifs/`。

输入应为 10 Hz、31 帧历史，数组长度与 `sim_duration` 一致。脚本保留历史和非预测车辆的数据，填充 `predict_mask` 指定车辆的未来轨迹。测试数据中 Ego 的未来无效，GIF 在历史结束后不再显示 Ego；模型内部仍会滚动预测所有车辆，包括 Ego。动图仅用于定性查看，不代表赛事评分。
