"""Example script using the mujoco python API. Originally based on the notebook
at
https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb.

"""

import time
from os import path

import mujoco
import mediapy as media
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path(path.join(path.dirname(__file__), "gymnasium_env/SO-ARM100/Simulation/SO101/scene.xml"))
data = mujoco.MjData(model)

print(len(data.ctrl))
print(len(data.qpos))
print(len(data.qvel))

with mujoco.Renderer(model) as renderer:
  mujoco.mj_forward(model, data)
  renderer.update_scene(data)

  plt.imshow(renderer.render())
  plt.show()
