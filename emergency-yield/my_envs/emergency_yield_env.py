from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from gymnasium.envs.registration import register

from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle


class EmergencyYieldEnv(HighwayEnv):
    """Ego should yield so an emergency vehicle behind can pass quickly & safely."""

    def __init__(self, config: dict | None = None, render_mode: str | None = None, **kwargs):
        # create attributes FIRST so reset() during super().__init__ can safely use them
        self.emergency = None
        self._gave_bonus = False
        super().__init__(config=config, render_mode=render_mode, **kwargs)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        cfg = super().default_config()
        cfg["right_lane_reward"] = 0.0
        cfg["high_speed_reward"] = 0.0
        cfg["collision_reward"]  = -1.0
        cfg["observation"].update({
            "type": "Kinematics",
            "vehicles_count": 15,
            "see_behind": True,
            "absolute": False,
            "normalize": True,
            "order": "sorted",
        })
        cfg.update({
            "speed_limit": 33.3,
            "success_gap": 2.0,
            "time_limit": 40.0,
            "penalties": {"tick_time": 0.005, "front_block": 0.02, "overspeed": 0.3},
            "emergency": {"spawn_dx": -60.0, "target_speed": 35.0, "siren_on": True},
            "siren_flash_hz": 4,   # <--- flash speed (times per second)
        })
        return cfg
    
    # emergency visual 
    def _update_emergency_visual(self) -> None:
        """Blink the emergency vehicle red/blue when siren is on."""
        if not self.config.get("siren_flash_hz", 0):
            return
        if not self.config["emergency"].get("siren_on", False):
            return
        if self.emergency is None:
            return
        # toggle color based on time
        phase = int(self.time * self.config["siren_flash_hz"]) % 2
        self.emergency.color = (220, 40, 40) if phase == 0 else (40, 120, 255)

    # ---- robust spawn path 1: called by AbstractEnv.reset() via HighwayEnv ----
    def _make_vehicles(self) -> None:
        super()._make_vehicles()          # ego + traffic
        self._add_emergency_vehicle()      # ensure emergency exists after base spawn

    # ---- robust spawn path 2: in case a wrapper bypasses _make_vehicles once ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        if self.emergency is None or self.emergency not in self.road.vehicles:
            self._add_emergency_vehicle()
        return obs, info

    def _add_emergency_vehicle(self) -> None:
        ego = self.vehicle
        lane = self.road.network.get_lane(ego.target_lane_index)
        x, _ = lane.local_coordinates(ego.position)
        ex = x + self.config["emergency"]["spawn_dx"]

        emergency = IDMVehicle.make_on_lane(
            self.road, ego.target_lane_index, longitudinal=ex, speed=30.0
        )
        emergency.target_speed = self.config["emergency"]["target_speed"]
        emergency.is_emergency = True

        self.road.vehicles.append(emergency)
        self.emergency = emergency
        self._gave_bonus = False

    # ---------- reward & termination ----------
    def _lane_width(self) -> float:
        return self.road.network.get_lane(self.vehicle.target_lane_index).width

    def _side_by_side(self) -> bool:
        if self.emergency is None:
            return False
        dx = abs(self.emergency.position[0] - self.vehicle.position[0])
        return dx <= self.config["success_gap"]

    def _reward(self, action: int) -> float:
        # If something weird happened and the emergency isn't there yet, return neutral reward
        self._update_emergency_visual()   # <-- add this line
        if self.emergency is None:
            return 0.0

        r_time = - self.config["penalties"]["tick_time"]

        dx = self.emergency.position[0] - self.vehicle.position[0]
        dy = abs(self.emergency.position[1] - self.vehicle.position[1])
        blocking = (dx < 0) and (dy < self._lane_width() / 2)
        r_block  = - self.config["penalties"]["front_block"] if blocking else 0.0

        v = self.vehicle.speed
        vmax = max(1e-6, self.config["speed_limit"])
        overspd = max(0.0, (v - vmax) / vmax)
        r_speed = - self.config["penalties"]["overspeed"] * overspd

        r_succ = 0.0
        if self._side_by_side() and not self._gave_bonus:
            r_succ = float(max(0.0, 1.0 - self.time / self.config["time_limit"]))
            self._gave_bonus = True

        return float(np.clip(r_time + r_block + r_speed + r_succ, 0.0, 1.0))

    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed or self._side_by_side())

    def _is_truncated(self) -> bool:
        return bool(self.time >= self.config["time_limit"])

# keep your register() call below this class
register(
    id="emergency-yield-v0",
    entry_point="my_envs.emergency_yield_env:EmergencyYieldEnv",
)

