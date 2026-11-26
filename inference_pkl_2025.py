"""Batch SMART inference for 2025 OnSite PKL + OpenDRIVE scenarios."""
import argparse
import os
import pickle
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

from inference import OnSiteSmartAgent, get_pkl_info, get_xodr_info, load_onsite_smart_model, DEFAULT_ONSITE_CKPT_PATH

HISTORY_FRAMES = 31


def infer_scene(exam, model):
    name = exam.name.removesuffix('_exam.pkl')
    xodr = exam.with_name(name + '.xodr')
    if not xodr.is_file():
        raise FileNotFoundError(xodr)
    with exam.open('rb') as f:
        scene = pickle.load(f)
    n, frames = scene['valid_mask'].shape
    duration = int(scene['sim_duration'])
    predict = np.asarray(scene['predict_mask'], dtype=bool)
    if scene['sim_freq'] != 10 or duration != frames or frames <= HISTORY_FRAMES:
        raise ValueError('expected 10 Hz and sim_duration frames with at least 31 history frames')
    if predict.shape != (n,) or scene['positions'].shape != (n, frames, 2) or scene['headings'].shape != (n, frames):
        raise ValueError('invalid agent array shapes')
    if not np.all(scene['valid_mask'][predict, HISTORY_FRAMES - 1]):
        raise ValueError('predicted agents must be valid at the last history frame')
    info = {'scenario_id': name, **get_xodr_info(str(xodr)), 'agent': get_pkl_info(str(exam))}
    with torch.inference_mode():
        result = OnSiteSmartAgent(info, model=model).inference(total_frame=duration)
    # Preserve all supplied history and every non-predicted agent, including Ego.
    for field, key in [('valid_mask', 'valid'), ('positions', 'position'), ('headings', 'heading')]:
        value = result[key].cpu().numpy()
        if value.shape[:2] != (n, frames) or (field != 'valid_mask' and not np.isfinite(value).all()):
            raise ValueError(f'invalid model output: {field} {value.shape}')
        scene[field] = np.array(scene[field], copy=True)
        scene[field][predict, HISTORY_FRAMES:] = value[predict, HISTORY_FRAMES:]
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test_dir', type=Path, default=Path('data/B'))
    parser.add_argument('--output_dir', type=Path, default=Path('results/pkl_2025'))
    parser.add_argument('--ckpt_path', type=Path, default=Path(DEFAULT_ONSITE_CKPT_PATH))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    exams = sorted(args.test_dir.rglob('*_exam.pkl'))
    if args.limit is not None:
        exams = exams[:args.limit]
    if not exams:
        parser.error(f'no exam PKL files found in {args.test_dir}')
    if len({p.name for p in exams}) != len(exams):
        parser.error('duplicate scenario names would overwrite output files')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Loading model: {args.ckpt_path}; scenarios: {len(exams)}', flush=True)
    model = load_onsite_smart_model(args.ckpt_path)
    failed = []
    for i, exam in enumerate(exams, 1):
        out = args.output_dir / exam.name.replace('_exam.pkl', '_output.pkl')
        if args.skip_existing and out.exists():
            print(f'[{i}/{len(exams)}] skip {out.name}', flush=True)
            continue
        print(f'[{i}/{len(exams)}] {exam.name}', flush=True)
        try:
            scene = infer_scene(exam, model)
            with tempfile.NamedTemporaryFile(dir=args.output_dir, delete=False) as f:
                tmp = f.name
                try:
                    pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)
                except BaseException:
                    os.unlink(tmp)
                    raise
            os.replace(tmp, out)
            print(f'  saved {out}', flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(str(exam))
    print(f'Finished: {len(exams)-len(failed)} succeeded or skipped; {len(failed)} failed', flush=True)
    if failed:
        print('\n'.join(failed), flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
