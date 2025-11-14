# train/train_dqn.py
import argparse, os
import gymnasium as gym
import highway_env
import my_envs.emergency_yield_env  # registers emergency-yield-v0
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

def make_env(obs_type: str):
    obs_cfgs = {
        "Kinematics": {"type":"Kinematics","vehicles_count":15,"see_behind":True,"absolute":False,"normalize":True,"order":"sorted"},
        "Lidar":      {"type":"LidarObservation","cells":64,"maximum_range":60,"normalize":True},
        "Grayscale":  {"type":"GrayscaleObservation","observation_shape":(128,64),"stack_size":4}
    }
    return gym.make("emergency-yield-v0", config={"observation": obs_cfgs[obs_type]}, render_mode=None)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["Kinematics","Lidar","Grayscale"], default="Kinematics")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--device", default="cpu")  # "cpu", "mps" (Apple), or "cuda"
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logdir", default="runs/dqn_emergency_yield")
    args = ap.parse_args()

    set_random_seed(args.seed)
    os.makedirs(args.logdir, exist_ok=True)

    env = make_env(args.obs)
    # Monitor writes episodic returns for the learning curve
    monitor_csv = os.path.join(args.logdir, f"monitor_{args.obs.lower()}.csv")
    env = Monitor(env, filename=monitor_csv)

    policy = "CnnPolicy" if args.obs == "Grayscale" else "MlpPolicy"

    model = DQN(
        policy, env,
        learning_rate=5e-4, buffer_size=50_000, batch_size=64,
        learning_starts=5_000, gamma=0.99, target_update_interval=500,
        train_freq=1, gradient_steps=1,
        tensorboard_log=args.logdir, verbose=1, device=args.device, seed=args.seed
    )

    eval_env = make_env(args.obs)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.logdir, "best", args.obs),
        log_path=args.logdir, eval_freq=10_000,
        deterministic=True, render=False, n_eval_episodes=10
    )

    model.learn(total_timesteps=args.steps, callback=eval_cb)
    model.save(os.path.join(args.logdir, f"dqn_{args.obs.lower()}"))
    print(f"[DONE] Trained {args.obs}. Monitor CSV: {monitor_csv}")
