import gymnasium as gym
import highway_env                  # registers built-in envs
import my_envs.emergency_yield_env  # registers "emergency-yield-v0"

# Headless smoke test
env = gym.make("emergency-yield-v0", config={
    "observation": {"type":"Kinematics","vehicles_count":15,"see_behind":True}
})

env = gym.make("emergency-yield-v0", render_mode="human", config={"offscreen_rendering": False})

obs, info = env.reset()

# (Optional sanity check)
assert hasattr(env.unwrapped, "emergency") and env.unwrapped.emergency is not None

done = truncated = False
while not (done or truncated):
    obs, r, done, truncated, info = env.step(1)  # IDLE
env.close()
print("Smoke test finished.")
