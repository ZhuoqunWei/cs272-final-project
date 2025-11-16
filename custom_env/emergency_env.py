from __future__ import annotations
import numpy as np
from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.highway_env import HighwayEnv
from gymnasium.envs.registration import register


class EmergencyVehicle(Vehicle):
    # emergency vehicle info (color, speed, different reward scales)
    info = {
        "police": {"color": (0, 0, 255), "speed": 30.0, "reward_scale": 1.0},
        "fire_truck": {"color": (255, 0, 0), "speed": 30.0, "reward_scale": 1.2},
        "ambulance": {"color": (255, 140, 0), "speed": 30.0, "reward_scale": 1.4},
    }
    # default emergency type is police
    def __init__(self, road, position, heading, emergency_type="police"):
        data = self.info[emergency_type]
        super().__init__(road, position, heading, data["speed"])
        self.is_emergency = True
        self.ev_type = emergency_type
        self.color = data["color"]
        self.target_speed = data["speed"]
        self.reward_scale = data["reward_scale"]

    # make sure emergency vehicle slows down if there is a car in front to prevent crashes
    def step(self, dt: float):
        super().step(dt)

        lane_index = self.lane_index[2]
        ego_position = self.position[0]

        # get all vehicles in the same lane ahead of this vehicle
        vehicles_in_lane = [
            v for v in self.road.vehicles
            if v.lane_index[2] == lane_index and v.position[0] > ego_position
        ]

        if vehicles_in_lane:
            front_vehicle = min(vehicles_in_lane, key=lambda v: v.position[0])
            distance = front_vehicle.position[0] - ego_position
            safe_distance = 7.0  # minimum distance to slow down to avoid crashes
            if distance < safe_distance:
                # slow down proportionally to distance
                self.speed = min(self.speed, front_vehicle.speed * (distance / safe_distance))
            else:
                # accelerate toward target speed
                self.speed = min(self.speed + 1.0 * dt, self.target_speed)
        else:
            # No car in front, accelerate toward target speed
            self.speed = min(self.speed + 1.0 * dt, self.target_speed)

class EmergencyEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                # same as default highway env config
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,  # The reward received when colliding with a vehicle.
                "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                # zero for other lanes.
                "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                # lower speeds according to config["reward_speed_range"].
                "lane_change_reward": 0,  # The reward received at each lane change action.
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return config

    def _create_vehicles(self) -> None:
        # same as highway env except adding emergency vehicle
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []

        # spawn ego vehicle same way as highway env
        for others in other_per_controlled:
            ego = Vehicle.create_random(
                self.road,
                speed=25.0,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            ego = self.action_type.vehicle_class(
                self.road, ego.position, ego.heading, ego.speed
            )
            self.controlled_vehicles.append(ego)
            self.road.vehicles.append(ego)

            # spawn emergency vehicle in a random lane but behind ego position
            rand_emergency_type = self.np_random.choice(list(EmergencyVehicle.info.keys()))
            
            # get all available lanes
            num_lanes = self.config["lanes_count"]
            emergency_lane_id = self.np_random.integers(0, num_lanes)
            emergency_lane = self.road.network.get_lane(("0", "1", emergency_lane_id))
            
            # spawn emergency vehicle behind ego (same x position offset, but in the random lane)
            ego_x = ego.position[0]
            ev_x = max(ego_x - 50, 0)  # don't go below 0
            
            emergency = EmergencyVehicle(
                road=self.road,
                position=emergency_lane.position(ev_x, 0),
                heading=emergency_lane.heading_at(ev_x),
                emergency_type=rand_emergency_type
            )
            emergency.lane_index = ("0", "1", emergency_lane_id)
            self.road.vehicles.append(emergency)

            # spawn regular vehicles like highway env
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road,
                    spacing=1 / self.config["vehicles_density"],
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        # new rewards for emergency vehicles
        yield_reward = 0.0
        ego_pos_x = self.vehicle.position[0]
        ego_speed = self.vehicle.speed
        ego_lane_index = self.vehicle.lane_index[2] # get lane number

        for other in self.road.vehicles:
            if getattr(other, "is_emergency", False):
                ev_pos_x = other.position[0]
                ev_speed = other.speed
                ev_lane_index = other.lane_index[2]

                if ev_pos_x < ego_pos_x and ev_lane_index == ego_lane_index:
                    # get distance and relative speed of emergency vehicle and ego car
                    distance = ego_pos_x - ev_pos_x
                    relative_speed = ev_speed - ego_speed

                    distance_factor = 1.0 / max(distance, 1.0)
                    speed_factor = np.clip(relative_speed / self.vehicle.target_speed, 0, 1)
                    urgency = distance_factor * (0.5 + 0.5 * speed_factor)

                    # reward for slowing and moving to the right
                    slow_factor = 1 - (ego_speed / self.vehicle.target_speed)
                    lane_factor = (max(ego_lane_index - 0, 0) / max(len(neighbours) - 1, 1))

                    reward_scale = other.reward_scale
                    yield_reward += reward_scale * urgency * (0.5 * slow_factor + 0.5 * lane_factor)

        rewards = {
            "collision_reward": self.config["collision_reward"] * float(self.vehicle.crashed),
            "right_lane_reward": self.config["right_lane_reward"] * (lane / max(len(neighbours) - 1, 1)),
            "high_speed_reward": self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "yield_emergency_reward": yield_reward
        }

        return rewards


# Register environment
register(
    id="EmergencyHighwayEnv-v0",
    entry_point="custom_env.emergency_env:EmergencyEnv",
)