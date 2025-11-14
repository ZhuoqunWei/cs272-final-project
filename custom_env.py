import gymnasium
import custom_env.emergency_env
import highway_env
import matplotlib
from matplotlib import pyplot as plt


env = gymnasium.make('EmergencyHighwayEnv-v0', render_mode='rgb_array')

env.reset()

for _ in range(3):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

# Show last frame
plt.imshow(env.render())
plt.axis("off")
plt.show()
