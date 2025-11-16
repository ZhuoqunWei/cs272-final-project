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
    """Emergency vehicle that maintains target speed but slows down if a car is in front."""
    def step(self, dt: float):
        super().step(dt)

        lane_index = self.lane_index[2]
        ego_position = self.position[0]

        # Get all vehicles in the same lane ahead of this vehicle
        vehicles_in_lane = [
            v for v in self.road.vehicles
            if v.lane_index[2] == lane_index and v.position[0] > ego_position
        ]

        if vehicles_in_lane:
            front_vehicle = min(vehicles_in_lane, key=lambda v: v.position[0])
            distance = front_vehicle.position[0] - ego_position
            safe_distance = 7.0  # minimum distance to maintain
            if distance < safe_distance:
                # Slow down proportionally to distance
                self.speed = min(self.speed, front_vehicle.speed * (distance / safe_distance))
            else:
                # Accelerate toward target speed
                self.speed = min(self.speed + 1.0 * dt, self.target_speed)
        else:
            # No car in front, accelerate toward target speed
            self.speed = min(self.speed + 1.0 * dt, self.target_speed)

class EmergencyEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "my_feature_enabled": True,
            "my_rew_scale": 2.0,
            "vehicles_count": 15,
        })
        return config

    def _create_road(self):
        # same road as highway-env
        super()._create_road()

    def _create_vehicles(self) -> None:
        # set lane for ego to spawn in
        ego_lane_id = 1

        # spawn ego vehicle in the specified lane
        ego_vehicle = Vehicle.create_random(
            self.road,
            speed=25.0,
            lane_id=ego_lane_id,
            spacing=self.config["ego_spacing"],
        )
        ego_vehicle = self.action_type.vehicle_class(
            self.road, ego_vehicle.position, ego_vehicle.heading, ego_vehicle.speed
        )
        self.controlled_vehicles = [ego_vehicle]
        self.vehicle = ego_vehicle
        ego_vehicle.color = (0, 255, 255) # make ego cyan
        self.road.vehicles.append(ego_vehicle)

        # define 3 different types of emergency vehicle (police/ambulance/fire truck)
        emergency_types = [
            {"name": "police", "color": (0, 0, 255), "speed": 27.0},
            {"name": "ambulance", "color": (255, 0, 0), "speed": 27.0},
            {"name": "fire_truck", "color": (255, 140, 0), "speed": 27.0},
        ]
        # choose a random one to spawn in the environment
        emergency = self.np_random.choice(emergency_types)

        # spawn emergency vehicle behind ego in the same lane
        ego_lane = self.road.network.get_lane(("0", "1", ego_lane_id))
        ego_position = ego_vehicle.position[0]
        emergency_position = max(ego_position - self.np_random.uniform(90, 100), 0)

        # create emergency vehicle that moves at a faster set speed
        emergency_vehicle = EmergencyVehicle(
            self.road,
            position=ego_lane.position(emergency_position, 0),
            speed=emergency["speed"],
            heading=ego_lane.heading_at(emergency_position),
        )

        emergency_vehicle.is_emergency = True
        emergency_vehicle.emergency_type = emergency["name"]
        emergency_vehicle.color = emergency["color"]
        emergency_vehicle.target_speed = emergency["speed"] # make sure it maintains the set speed
        self.road.vehicles.append(emergency_vehicle)

        # spawn regular vehicles in other lanes
        num_lanes = self.config["lanes_count"]
        other_lane_ids = [lane_id for lane_id in range(num_lanes) if lane_id != ego_lane_id]

        # calculate how many regular vehicles to spawn in other lanes
        vehicles_per_lane = near_split(self.config["vehicles_count"], num_bins=len(other_lane_ids))
        vehicles_per_lane = [int(v) for v in vehicles_per_lane]

        for lane_id, num_vehicles in zip(other_lane_ids, vehicles_per_lane):
            lane = self.road.network.get_lane(("0", "1", lane_id))

            # spawn regular vehicles in a range around the ego vehicle
            spawn_range = 500
            min_x = max(50, ego_position - spawn_range / 2)
            max_x = min(lane.length - 50, ego_position + spawn_range / 2)

            # make sure regular vehicles are not too close to each other
            min_spacing = 20.0  # minimum distance between vehicles
            x_positions = []

            for _ in range(num_vehicles):
                for attempt in range(100):  # max 100 tries
                    candidate_x = self.np_random.uniform(min_x, max_x)
                    if all(abs(candidate_x - x) >= min_spacing for x in x_positions):
                        x_positions.append(candidate_x)
                        break

            x_positions = np.sort(x_positions)

            for x in x_positions:
                regular_vehicles = Vehicle(
                    self.road,
                    position=lane.position(x, 0),
                    heading=lane.heading_at(x),
                    speed=24.0
                )
                self.road.vehicles.append(regular_vehicles)

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

        # custom reward scale for the different emergency vehicle types
        emergency_reward_scale = {
            "police": 1.0,
            "ambulance": 1.5,
            "fire_truck": 2.0
        }

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

                    reward_scale = emergency_reward_scale.get(getattr(other, "ev_type", "police"), 1.0)
                    yield_reward += reward_scale * urgency * (0.5 * slow_factor + 0.5 * lane_factor)

        rewards = {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "yield_emergency_reward": yield_reward
        }

        return rewards


# Register environment
register(
    id="EmergencyHighwayEnv-v0",
    entry_point="custom_env.emergency_env:EmergencyEnv",
)