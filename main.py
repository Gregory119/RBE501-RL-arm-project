import gymnasium as gym
import gymnasium_env

from stable_baselines3 import PPO

env = gym.make('Arm-v0',render_mode='human')

model = PPO("MlpPolicy",env,device='cpu')
model.learn(total_timesteps=1000,progress_bar=True)

# evaluate model after training
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, state = model.predict(obs)  # agent policy that uses the observation and info
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

model.close()
