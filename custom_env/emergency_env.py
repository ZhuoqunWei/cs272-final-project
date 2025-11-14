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

class EmergencyEnv(HighwayEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "my_feature_enabled": True,
            "my_rew_scale": 2.0,
        })
        return config

    def _create_road(self):
        """Use EXACT SAME road as highway-v0."""
        super()._create_road()

    def _create_vehicles(self):
        super()._create_vehicles()
        # choose lane to spawn emergency vehicle
        lane_index = 1
        lane = self.road.network.get_lane(("0", "1", lane_index))
        # want to spawn behind ego car, so get it's position
        ego = self.vehicle
        ego_pos = ego.position[0]  # ego horizontal position

        # spawn emergency vehicle 20 meters behind ego
        ev_position = max(ego_pos - 20, 0)  # can't go negative position

        # Create emergency vehicle
        ev = self.action_type.vehicle_class(
            road=self.road,
            position=lane.position(ev_position, 0),
            heading=0,
            speed=35  # make it faster than regular vehicles
        )
        ev.is_emergency = True
        ev.color = (255, 165, 0)  # make emergency vehicle orange

        self.road.vehicles.append(ev)

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
        # default rewards without emergency vehicle (same as default)
        rewards = {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }
        # new rewards for emergency vehicles
        yield_reward = 0.0
        ego_pos = self.vehicle.position
        ego_lane_index = self.vehicle.lane_index[2] # get lane number

        for other in self.road.vehicles:
            if getattr(other, "is_emergency", False):
                # Check if emergency vehicle is behind ego and in same lane
                other_pos = other.position
                other_lane_index = other.lane_index[2]
                if other_pos[0] < ego_pos[0] and other_lane_index == ego_lane_index:
                    # reward for slowing down and moving right
                    speed_factor = 1 - (forward_speed / self.vehicle.target_speed)
                    lane_factor = (max(ego_lane_index - 0, 0) / max(len(neighbours) - 1, 1))  # closer to right lane
                    yield_reward += 0.5 * speed_factor + 0.5 * lane_factor

        rewards["yield_emergency_reward"] = yield_reward

        return rewards


# Register environment
register(
    id="EmergencyHighwayEnv-v0",
    entry_point="custom_env.emergency_env:EmergencyEnv",
)