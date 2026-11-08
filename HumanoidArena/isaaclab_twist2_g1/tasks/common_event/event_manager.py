# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
event manager
"""
import torch
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as base_mdp


class SimpleEvent:
    def __init__(self, func, params=None):
        self.func = func
        self.params = params or {}

    def trigger(self, env):
        return self.func(env, **self.params)


class MultiObjectEvent:
    """Event class supporting operations on multiple objects."""
    
    def __init__(self, reset_configs=None):
        """
        Args:
            reset_configs: List of object reset configurations in the following format:
            [
                {
                    "asset_cfg": SceneEntityCfg("red_block"),
                    "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
                    "velocity_range": {}
                },
                {
                    "asset_cfg": SceneEntityCfg("yellow_block"),
                    "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03]},
                    "velocity_range": {}
                }
            ]
        """
        self.reset_configs = reset_configs or []
    
    def trigger(self, env):
        """
        Trigger a reset event for multiple objects.
        """
        env_ids = torch.arange(env.num_envs, device=env.device)
        results = []
        
        for config in self.reset_configs:
            try:
                result = base_mdp.reset_root_state_uniform(
                    env=env,
                    env_ids=env_ids,
                    pose_range=config.get("pose_range", {}),
                    velocity_range=config.get("velocity_range", {}),
                    asset_cfg=config["asset_cfg"]
                )
                results.append(result)
                print(f"Resetting object: {config['asset_cfg'].name}")
            except Exception as e:
                print(f"Failed to reset object {config['asset_cfg'].name}: {e}")
        
        return results


class BatchObjectEvent:
    """Batch object event class with flexible configuration."""
    
    def __init__(self, object_names=None, pose_ranges=None, velocity_ranges=None):
        """
        Args:
            object_names: Object names, such as ["red_block", "yellow_block", "green_block"]
            pose_ranges: Position ranges specified as either:
                - One configuration shared by all objects: {"x": [-0.05, 0.05], "y": [-0.05, 0.05]}
                - A dictionary with per-object ranges: {"red_block": {...}, "yellow_block": {...}}
            velocity_ranges: Velocity ranges, using the same format as pose_ranges.
        """
        self.object_names = object_names or []
        self.pose_ranges = pose_ranges or {}
        self.velocity_ranges = velocity_ranges or {}
    
    def trigger(self, env):
        """Trigger a batch reset."""
        env_ids = torch.arange(env.num_envs, device=env.device)
        results = []
        
        for obj_name in self.object_names:
            try:
                # Get this object's pose_range configuration
                if isinstance(self.pose_ranges, dict) and obj_name in self.pose_ranges:
                    pose_range = self.pose_ranges[obj_name]
                elif isinstance(self.pose_ranges, dict) and "x" in self.pose_ranges:
                    # Single configuration shared by all objects
                    pose_range = self.pose_ranges
                else:
                    pose_range = {}
                
                # Get this object's velocity_range configuration
                if isinstance(self.velocity_ranges, dict) and obj_name in self.velocity_ranges:
                    velocity_range = self.velocity_ranges[obj_name]
                elif isinstance(self.velocity_ranges, dict) and "linear" in self.velocity_ranges:
                    # Single configuration shared by all objects
                    velocity_range = self.velocity_ranges
                else:
                    velocity_range = {}
                
                result = base_mdp.reset_root_state_uniform(
                    env=env,
                    env_ids=env_ids,
                    pose_range=pose_range,
                    velocity_range=velocity_range,
                    asset_cfg=SceneEntityCfg(obj_name)
                )
                results.append(result)
                print(f"✅ Reset object: {obj_name}")
                
            except Exception as e:
                print(f"❌ Failed to reset object {obj_name}: {e}")
        
        return results


class SimpleEventManager:
    def __init__(self):
        self._events = {}

    def register(self, name, event):
        self._events[name] = event

    def unregister(self, name):
        """Unregister an event if it exists."""
        self._events.pop(name, None)

    def trigger(self, name, env):
        event = self._events.get(name)
        if event:
            return event.trigger(env)
        else:
            print(f"Event {name} not registered")
    
    def register_multi_object_reset(self, name, object_names, pose_ranges=None, velocity_ranges=None):
        """
        Convenience method for registering a multi-object reset event.
        
        Args:
            name: Event name.
            object_names: List of object names.
            pose_ranges: Position range configuration.
            velocity_ranges: Velocity range configuration.
        """
        event = BatchObjectEvent(
            object_names=object_names,
            pose_ranges=pose_ranges,
            velocity_ranges=velocity_ranges
        )
        self.register(name, event)