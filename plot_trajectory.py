import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt


# return trajectory and goal from the file
def load_traj_and_goal(path: pathlib.Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        traj = data["traj"]
        goal = data["goal"]
    else:  # plain .npy
        traj = data
        goal = None

    if traj.ndim != 2 or traj.shape[1] != 3:
        raise ValueError(f"{path}: expected shape (N,3); got {traj.shape}")
    return traj, goal



def main(paths):
    if not 1 <= len(paths) <= 2:
        print(__doc__)
        sys.exit(1)

    traj_goal_pairs = [load_traj_and_goal(p) for p in paths]
    trajs = [tg[0] for tg in traj_goal_pairs]
    goal  = next((tg[1] for tg in traj_goal_pairs if tg[1] is not None), None)

    
    # initialize figure
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")

    colors = ["b", "g"] # blue for trajectory 1, green for trajectory 2
    labels = ["unscaled", "scale = 9.0"]

    # plot trajectories
    for i, traj in enumerate(trajs):
        ax.plot3D(
            traj[:, 0], traj[:, 1], traj[:, 2],
            lw=2, color=colors[i], label=labels[i]
        )

    # goal point
    if goal is not None:
        ax.scatter(goal[0], goal[1], goal[2], color="r", s=40, label="goal")


    all_points = np.vstack(trajs)
    max_range  = (all_points.max(0) - all_points.min(0)).max()
    mid        = all_points.mean(0)
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3-D Trajectory Comparison")
    ax.legend()


    # Compute and display errors
    if goal is not None:
        errors = [np.linalg.norm(traj[-1] - goal) for traj in trajs]
        error_lines = [
            f"{labels[i]} error: {errors[i]:.3f} m" for i in range(len(errors))
        ]
        # place the text
        fig.text(
            0.02, 0.02, "\n".join(error_lines),
            ha="left", va="bottom", fontsize=10, family="monospace"
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main([pathlib.Path(p).expanduser() for p in sys.argv[1:]])