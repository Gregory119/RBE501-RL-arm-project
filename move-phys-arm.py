"""
Simple script to control the physical follower arm

Example:

```shell
python move-phys-arm.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm \
```
"""

import logging
import mujoco
import time
from os import path
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np

from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun



@dataclass
class MoveConfig:
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60


def inbounds(robot: Robot, bounds: dict[str,float]):
    #print("inbounds()")
    # get current joint positions
    present_pos = robot.bus.sync_read("Present_Position")
    pos_deg = np.array([p for _,p in present_pos.items()])
    #print("qpos_deg: {}".format(pos_deg))
    qpos = pos_deg / 180 * np.pi
    #print("qpos: {}".format(qpos))

    # calculate forward kinematics
    xml_file = path.join(
        path.dirname(__file__),
        "gymnasium_env/SO-ARM100/Simulation/SO101/scene.xml"
        )
    model = mujoco.MjModel.from_xml_path(xml_file)
    data = mujoco.MjData(model)
    data.qpos = qpos
    mujoco.mj_forward(model,data)

    ee_pos = data.site("gripper").xpos
    #print("ee_pos: {}".format(ee_pos))

    # convert ee from cartesian to cyclindrical coordinates for bounds check
    x,y,z = ee_pos
    ee_phi = np.arctan2(y,x)
    ee_rho = np.linalg.norm([x,y])
    ee_z = z
    ee_cyl = np.array([ee_rho, ee_phi, ee_z])
    #print("ee_cyl: {}".format(ee_cyl))
    
    # check if ee in bounds
    return np.all(ee_cyl > bounds['lower']) and np.all(ee_cyl < bounds['upper'])
    

def resetPos(robot: Robot, rate: int):
    goal_pos = {'shoulder_pan.pos': 0, 'shoulder_lift.pos': -80.0, 'elbow_flex.pos': 90, 'wrist_flex.pos': 0, 'wrist_roll.pos': 0, 'gripper.pos': 0}
    goal_pos_vec = np.array([p for _,p in goal_pos.items()])
    robot.send_action(goal_pos)

    # wait for robot to start moving
    busy_wait(1)

    # wait for robot to reach position (stop moving)
    goal_tol = 0.01
    while True:
        # By default the robot doesn't use integral gain so the joint position
        # error can be large and be different based on the robot
        # configuration. Using joint position errors to check for reaching the
        # goal is therefore unreliable. Instead wait for the joint velocity to
        # reach close to zero after initially moving.
        
        present_vel = robot.bus.sync_read("Present_Velocity")
        vel = np.array([v for _,v in present_vel.items()])
        #print("vel {}".format(vel))
        vel_norm = np.linalg.norm(vel)
        print("vel_norm: {}".format(vel_norm))
        if vel_norm < goal_tol:
            break
        # the wait duration doesn't have to be perfect so ignore duration of
        # sync_read()
        busy_wait(1/rate)
    print("position reset")
        

def run_loop(robot: Robot, bounds: dict[str,float], fps: int, action: dict[str,int]):
    resetPos(robot,fps)

    start = time.perf_counter()
    fault = False
    while True:
        loop_start = time.perf_counter()

        # if ee not in bounds, then reset position
        out_of_bounds = not inbounds(robot, bounds)
        if not fault and out_of_bounds:
            print('fault triggered')
            fault = True
            resetPos(robot,fps)
        elif fault:
            pass
        else:
            #print("action: {}".format(action))
            robot.send_action(action)

        # sleep remaining loop duration
        dt_s = time.perf_counter() - loop_start
        if dt_s <0:
            print("failed to complete commands in loop duration")
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start


def createBounds():
    # set limits in cylindrical coordinates
    # rho, phi, z
    lower = np.array([0.1143,-np.pi,0.075])
    upper = np.array([0.4064,np.pi,0.25])
    return {'lower': lower, 'upper':upper}
        
@draccus.wrap()
def testMove(cfg: MoveConfig):
    cfg.robot.use_degrees = True

    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    bounds = createBounds()
    action = {'shoulder_pan.pos': 0, 'shoulder_lift.pos': 0.0, 'elbow_flex.pos': 0, 'wrist_flex.pos': 0, 'wrist_roll.pos': 0, 'gripper.pos': 0}

    try:
        run_loop(robot, bounds, cfg.fps, action)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    testMove()
