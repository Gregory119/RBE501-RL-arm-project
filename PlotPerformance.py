#!/usr/bin/env python3

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LABEL_MAP = {
    "PPO_model_3_run_1": "0.5",
    "PPO_model_3_run_2": "1.0",
    "PPO_model_3_run_3": "1.7",
    "PPO_model_3_run_4": "2.3",
    "PPO_model_3_run_5": "4.44",
    "PPO_model_3_run_6": "9.0",
   
}


LEGEND_TITLE = "Link Mass/Inertia Scaling Factor"
LEGEND_LOC = "center left"


#return steps and values for a single scalar tag inside the logdir
def load_scalar(logdir: Path, tag: str):
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise KeyError(
            f"Scalar tag '{tag}' not found in {logdir}. "
            f"Available: {ea.Tags()['scalars']}"
        )

    events = ea.Scalars(tag)
    steps  = np.fromiter((e.step  for e in events), dtype=np.int64)
    values = np.fromiter((e.value for e in events), dtype=np.float64)
    return steps, values

#smoothing
def smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    kernel = np.ones(k) / k
    return np.convolve(arr, kernel, mode="valid")



def main():
    parser = argparse.ArgumentParser(
        description="Convert TensorBoard scalars to Matplotlib figures"
    )
    parser.add_argument(
        "--logdirs",
        nargs="+",
        required=True,
        help="One or more paths containing TensorBoard event files.",
    )
    parser.add_argument(
        "--scalars",
        nargs="+",
        default=["episode/reward"],
        help="Scalar tags to plot (space-separated list).",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=1,
        help="Moving-average window (in timesteps) for smoothing curves.",
    )
    parser.add_argument(
        "--outfig",
        default="tb_plot.png",
        help="Destination file (extension chooses format: .png, .pdf, …).",
    )
    parser.add_argument(
        "--legend", action="store_true", help="Show a legend (one curve per logdir)."
    )
    args = parser.parse_args()

    # matplotlib
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": 300,
        }
    )

    n_rows = len(args.scalars)
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(6, 3.2 * n_rows), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    # draw every requested run and scalar 
    for logdir in args.logdirs:
        run_name = Path(logdir).stem
        pretty   = LABEL_MAP.get(run_name, run_name)
        for ax, tag in zip(axes, args.scalars):
            steps, vals = load_scalar(Path(logdir), tag)
            vals = smooth(vals, args.smoothing)
            steps = steps[: len(vals)]  # keep arrays aligned after smoothing
            ax.plot(steps, vals, label=pretty, linewidth = 1.0)

            # format axes
            ax.set_ylabel(tag.replace("/", "\n"))
            if args.legend:
                ax.legend(loc = LEGEND_LOC,
                    framealpha=0.8, 
                    title = LEGEND_TITLE,
                    prop={"size": 8},
                    markerscale=0.8,
                    handlelength=1.5,
                    handletextpad=0.2,
                    borderpad=0.3,)

    axes[-1].set_xlabel("environment steps")
    fig.suptitle("Evaluation in Simulation")

    fig.savefig(args.outfig, bbox_inches="tight")
    print(f"saved figure to {args.outfig!s}")


if __name__ == "__main__":
    main()