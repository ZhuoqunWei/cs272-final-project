# train/eval_500.py
import argparse, numpy as np, pandas as pd
import gymnasium as gym
import highway_env
import my_envs.emergency_yield_env
from stable_baselines3 import DQN

def make_env(obs_type):
    cfg = {
        "Kinematics": {"type":"Kinematics","vehicles_count":15,"see_behind":True,"absolute":False,"normalize":True,"order":"sorted"},
        "Lidar":      {"type":"LidarObservation","cells":64,"maximum_range":60,"normalize":True},
        "Grayscale":  {"type":"GrayscaleObservation","observation_shape":(128,64),"stack_size":4}
    }[obs_type]
    return gym.make("emergency-yield-v0", config={"observation": cfg})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, choices=["Kinematics","Lidar","Grayscale"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--out", default="eval_returns.csv")
    args = ap.parse_args()

    env = make_env(args.obs)
    model = DQN.load(args.model_path, device="cpu")

    rows = []
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = truncated = False
        ep_ret = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, info = env.step(int(action))
            ep_ret += r
        # success = terminated by _side_by_side (not by crash) → vehicle.crashed == False
        crashed = bool(env.unwrapped.vehicle.crashed)
        success = (done and not crashed)
        time_to_clear = float(env.unwrapped.time) if success else np.nan
        rows.append({"return": ep_ret, "success": success, "crashed": crashed, "time_to_clear": time_to_clear})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} | mean return={df['return'].mean():.3f} | "
          f"success rate={(df['success'].mean()*100):.1f}% | "
          f"mean time_to_clear={df['time_to_clear'].mean():.2f}s")
