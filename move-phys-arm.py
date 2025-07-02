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
import signal
from os import path
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
from types import FunctionType

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
    """Input configuration to this script which can be set using
    commandline arguments."""
    robot: RobotConfig
    # Limit the maximum frames per second.
    rate: int = 60

@dataclass
class CylCoord:
    """Represents cylindrical coordinates."""
    rho: float = 0.0
    phi: float = 0.0
    z: float = 0.0

    def __lt__(self, other):
        if not isinstance(other, CylCoord):
            return NotImplemented

        return self.rho < other.rho and self.phi < other.phi and self.z < other.z

    def __gt__(self, other):
        if not isinstance(other, CylCoord):
            return NotImplemented

        return self.rho > other.rho and self.phi > other.phi and self.z > other.z

    
@dataclass
class RobotBounds:
    lower: CylCoord
    upper: CylCoord


def sig_handler(sig, frame):
    global g_stop
    if sig == signal.SIGINT: # Ctrl-C
        g_stop = True


def in_bounds(robot: Robot, bounds: RobotBounds) -> bool:
    """
    Check if the robot is in bounds.
    
    @param bounds The bounds in cylindrical coordinates.
    @return True if the robot is in bounds, otherwise false.
    """
    
    #print("in_bounds()")
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
    ee_cyl = CylCoord(rho=ee_rho, phi=ee_phi, z=ee_z)
    #print("ee_cyl: {}".format(ee_cyl))
    
    # check if ee in bounds
    return np.all(ee_cyl > bounds.lower) and np.all(ee_cyl < bounds.upper)
    

def reset_pos(robot: Robot, rate: int):
    """Move the robot to a default start configuration."""

    q_goal_dict = {'shoulder_pan.pos': 0, 'shoulder_lift.pos': -80.0, 'elbow_flex.pos': 90, 'wrist_flex.pos': 0, 'wrist_roll.pos': 0, 'gripper.pos': 0}
    robot.send_action(q_goal_dict)

    # By default the robot doesn't use integral gain so the joint position
    # error can be large and depend on the robot configuration. Using joint
    # position errors to check for reaching the goal is therefore
    # unreliable. Instead wait for the joint velocity to reach close to
    # zero after initially moving.

    # wait for robot to start moving
    busy_wait(1)

    # wait for robot to reach position (stop moving)
    dq_norm_tol = 0.01
    while True:    
        dq_dict = robot.bus.sync_read("Present_Velocity")
        dq = np.array([v for _,v in dq_dict.items()])
        #print("dq {}".format(dq))
        dq_norm = np.linalg.norm(dq)
        print("dq_norm: {}".format(dq_norm))
        if dq_norm < dq_norm_tol:
            break
        # the wait duration doesn't have to be perfect so ignore duration of
        # sync_read()
        busy_wait(1/rate)
    print("position reset")
        

def run_loop(robot: Robot, bounds: RobotBounds, rate: int, action: dict[str,int], stop_fn: FunctionType):
    reset_pos(robot,rate)

    start = time.perf_counter()
    fault = False
    while not stop_fn():
        loop_start = time.perf_counter()

        out_of_bounds = not in_bounds(robot, bounds)
        if not fault and out_of_bounds:
            # if ee not in bounds, then reset position and fault
            print('fault triggered')
            fault = True
            reset_pos(robot,rate)
        elif fault:
            pass
        else:
            #print("action: {}".format(action))
            robot.send_action(action)

        # sleep remaining loop duration
        dt_s = time.perf_counter() - loop_start
        if dt_s < 0:
            print("failed to complete commands in loop duration")
        busy_wait(1 / rate - dt_s)

        loop_s = time.perf_counter() - loop_start

    reset_pos(robot,rate)
        

def createBounds():
    # set limits in cylindrical coordinates
    # rho, phi, z
    lower = CylCoord(rho=0.1143,phi=-np.pi,z=0.075)
    upper = CylCoord(rho=0.4064,phi=0,z=0.25)
    return RobotBounds(lower=lower, upper=upper)


g_stop = False

@draccus.wrap()
def testMove(cfg: MoveConfig):
    signal.signal(signal.SIGINT, sig_handler)
    
    cfg.robot.use_degrees = True

    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    bounds = createBounds()
    action = {'shoulder_pan.pos': 0, 'shoulder_lift.pos': 0.0, 'elbow_flex.pos': 0, 'wrist_flex.pos': 0, 'wrist_roll.pos': 0, 'gripper.pos': 0}
    stop_fn = lambda: g_stop

    try:
        run_loop(robot, bounds, cfg.rate, action, stop_fn)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    testMove()
