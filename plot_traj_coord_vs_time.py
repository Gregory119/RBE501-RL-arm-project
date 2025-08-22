
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt

'''
This script plots each coordinate component of a trajectory or trajectories along with the goal against the evaluation step. 
To run the script add the trajectory file name(with relative path) in the command line. Legend labels are hard coded. The output plot is displayed and must 
be manually saved
'''

#read file data
def load_traj_goal(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        return data["traj"], data["goal"]
    return data, None


# plot
def plot_component(comp_list, goal_val, coord, colours, labels=None):
    
    if labels is None:
        labels = [f"trajectory {i + 1}" for i in range(len(comp_list))]

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, comp in enumerate(comp_list):
        ax.plot(comp, color=colours[i], label=labels[i])

    if goal_val is not None:
        ax.axhline(goal_val, color="r", ls="--", label="goal")

    ax.set_xlabel("Step")
    ax.set_ylabel(f"{coord}-position [m]")
    ax.set_title(f"{coord.upper()}-coordinate vs. Step")
    ax.legend()
    fig.tight_layout()



def main(paths):
    if not 1 <= len(paths) <= 2:
        print(__doc__)
        sys.exit(1)

    trajs, goals = [], []
    for p in paths:
        traj, goal = load_traj_goal(p)
        if traj.ndim != 2 or traj.shape[1] != 3:
            raise ValueError(f"{p}: expected shape (N,3); got {traj.shape}")
        trajs.append(traj)
        goals.append(goal)

    
    goal = next((g for g in goals if g is not None), None)
    goal_x = goal[0] if goal is not None else None
    goal_y = goal[1] if goal is not None else None
    goal_z = goal[2] if goal is not None else None

    # split into components
    x_list = [t[:, 0] for t in trajs]
    y_list = [t[:, 1] for t in trajs]
    z_list = [t[:, 2] for t in trajs]

    traj_colours = ["b", "g"] # blue for trajectory 1, green for trajectory 2
    labels       = ["unscaled", "scale = 9.0"]

    plot_component(x_list, goal_x, "x", traj_colours[: len(x_list)], labels[: len(x_list)])
    plot_component(y_list, goal_y, "y", traj_colours[: len(y_list)], labels[: len(y_list)])
    plot_component(z_list, goal_z, "z", traj_colours[: len(z_list)], labels[: len(z_list)])

    plt.show()


if __name__ == "__main__":
    main([pathlib.Path(p).expanduser() for p in sys.argv[1:]])
