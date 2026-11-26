"""Animate inferred PKL scenes using the project's SMART map and vehicle renderer."""
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon
import numpy as np

from inference import OnSiteSmartAgent, get_xodr_info


def render(path, data_dir, output_dir):
    name = path.name.removesuffix('_output.pkl')
    with path.open('rb') as f:
        scene = pickle.load(f)
    map_data = {'scenario_id': name, **get_xodr_info(str(data_dir / name / (name + '.xodr')))}
    valid, pos, headings = scene['valid_mask'], scene['positions'], scene['headings']
    fig, ax = plt.subplots(figsize=(10, 7))
    OnSiteSmartAgent.plot_static_map(OnSiteSmartAgent.__new__(OnSiteSmartAgent), ax, map_data)
    xy = pos[valid]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = max(20, (hi - lo).max() * .08)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect('equal', adjustable='box')
    patches = []
    for i, ident in enumerate(scene['ids']):
        patch = Polygon(np.zeros((4, 2)), closed=True, facecolor='#e34a33' if ident == 'Ego' else '#3182bd', edgecolor='black', linewidth=.35, zorder=3)
        ax.add_patch(patch)
        patches.append(patch)
    time_text = ax.text(.02, .98, '', transform=ax.transAxes, va='top', bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    fig.tight_layout()

    def update(frame):
        for i, patch in enumerate(patches):
            patch.set_visible(bool(valid[i, frame]))
            if valid[i, frame]:
                x, y = pos[i, frame]
                length, width = scene['shapes'][i]
                patch.set_xy(OnSiteSmartAgent._cal_polygon_contour(np.array([x]), np.array([y]), np.array([headings[i, frame]]), np.array([width]), np.array([length]))[0])
        time_text.set_text(f"{frame / scene['sim_freq']:.1f} s  |  " + ('history' if frame < 31 else 'prediction'))
        return [*patches, time_text]

    anim = FuncAnimation(fig, update, frames=pos.shape[1], blit=True)
    anim.save(output_dir / (name + '.gif'), writer=PillowWriter(fps=scene['sim_freq']), dpi=75)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path('results/pkl_2025'))
    parser.add_argument('--data_dir', type=Path, default=Path('data/B'))
    parser.add_argument('--gif_dir', type=Path, default=Path('results/pkl_2025/gifs'))
    args = parser.parse_args()
    args.gif_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.output_dir.glob('*_output.pkl')):
        render(path, args.data_dir, args.gif_dir)
        print(f'saved {path.stem}', flush=True)


if __name__ == '__main__':
    main()
