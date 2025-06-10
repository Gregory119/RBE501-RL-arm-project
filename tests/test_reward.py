# tests/test_reward.py
from __future__ import annotations
import os
import numpy as np
import mujoco
import pytest

from gymnasium_env.arm import ArmEnv



_TEST_DIR  = os.path.dirname(__file__)
DUMMY_XML  = os.path.join(_TEST_DIR, "dummy.xml")


def site_id(model, name: str) -> int:
    
    if hasattr(model, "site_name2id"):
        return model.site_name2id(name)
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)



@pytest.mark.parametrize(
    "offset, expected, atol",
    [
        # (offset vector, expected reward, tolerance)
        (np.zeros(3),               5.0, 0.01),   # at goal
        (np.array([0.1, 0, 0]),    -0.1, 0.02),   # 10 away
    ],
)
def test_reward_distance_cases(offset: np.ndarray, expected: float, atol: float):
    # headless env
    env = ArmEnv(xml_file=DUMMY_XML, render_mode=None)
    env.reset(seed=0)

    sid = site_id(env.model, "ee_site")
    env.goal = env.data.site_xpos[sid] + offset

   
    _, reward, *_ = env.step(np.zeros(env.action_space.shape))

    
    assert np.isclose(reward, expected, atol=atol), (
        f"Reward {reward:.3f} differs from expected {expected}"
    )

    env.close()