# CS272_EmergencyHighway_Env

## 1. Overview

`EmergencyHighwayEnv-v0` is based on `highway-env` but introduces behavior for **emergency vehicles (EVs)** and corresponding rewards for the **ego vehicle**.  

- Multi-lane traffic with both regular and emergency vehicles.  
- Emergency vehicles (police, fire truck, ambulance) spawn in middle lanes.  
- Each EV class has unique attributes like speed, reward scaling, and color.  
- Ego vehicle must yield appropriately.  
- Side-lane vehicles are passive.

---

## 2. Objective

The agent (ego vehicle) must:

- Drive efficiently through traffic while maintaining road safety.  
- Avoid collisions with all vehicles.  
- Yield properly and adjust speed when an emergency vehicle approaches from behind.  
- Stay primarily in the right lanes.  

---

## 3. Action Space

DiscreteMetaActions same as `highway-env` .

```text
0 → LANE_LEFT
1 → IDLE
2 → LANE_RIGHT
3 → FASTER
4 → SLOWER
```

---

## 4. Observation Space

Kinematics same as `highway-env` .

| Index | Feature | Meaning |
|--------|----------|---------|
| 0 | `presence` | 1.0 if this slot is filled by a vehicle|
| 1 | `x` | Longitudinal position |
| 2 | `y` | Lateral position offset|
| 3 | `vx` | Longitudinal velocity component |
| 4 | `vy` | Lateral velocity component |

---

## 5. Emergency Vehicle Types

The environment defines an `EmergencyVehicle` class with three variants:

| Type | Color (RGB) | Default Speed (m/s) | Reward Scale |
|------|--------------|---------------------|---------------|
| Police | (0, 0, 255) | 30.0 | 1.0 |
| Fire Truck | (255, 0, 0) | 30.0 | 1.2 |
| Ambulance | (255, 140, 0) | 30.0 | 1.4 |

---

## 6. Reward Function

| Reward Component | Description |
|------------------|-------------|
| `collision_reward` | Negative reward on collisions. |
| `right_lane_reward` | Bonus for driving in right-most lanes. |
| `high_speed_reward` | Positive scale for maintaining target speed. |
| `yield_emergency_reward` | Positive reward for proper yielding or slowing near EVs. |
| `on_road_reward` | Positive reward for staying on the roadway. |

---

## 7. Termination Conditions

The episode terminates on:

- **Success:** Scenario completion.  
- **Collision:** Any contact with another vehicle or boundary.  
- **Timeout:** Maximum step duration reached (default 40s).

---

## 8. Usage

### Register the Environment

```python
import custom_env.emergency_env
```

### Create Environment

```python
import gymnasium as gym
env = gym.make("EmergencyHighwayEnv-v0")
```

### With Rendering

```python
env = gym.make("EmergencyHighwayEnv-v0", render_mode="human")
```

### Run Example Loop

```python
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
```

---

## 9. Environment Registration

```python
register(
    id="EmergencyHighwayEnv-v0",
    entry_point="custom_env.emergency_env:EmergencyEnv",
)
```
