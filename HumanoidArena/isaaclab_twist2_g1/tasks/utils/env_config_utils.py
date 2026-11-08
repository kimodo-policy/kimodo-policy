"""
Environment configuration utilities.
Provides enhanced environment configuration management and argument handling.
"""

import argparse
from typing import Dict, Any, Optional
from pathlib import Path


def create_enhanced_env_cfg(task_name: str, args: argparse.Namespace):
    """Create an enhanced environment configuration.
    
    Args:
        task_name: Task name.
        args: Command-line arguments.
        
    Returns:
        Environment configuration object.
    """
    
    # Choose the configuration factory for the task
    if "Isaac-PickPlace-G129" in task_name:
        return create_g129_pickplace_cfg(args)
    else:
        # Fall back to the standard factory
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        return parse_env_cfg(task_name, device=args.device, num_envs=getattr(args, 'num_envs', 1))


def create_g129_pickplace_cfg(args: argparse.Namespace):
    """Create the environment configuration for the G129 pick-and-place task.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        PickPlaceG129DEX1JointWaistFixEnvCfg: Environment configuration.
    """
    from tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex1.pickplace_cylinder_g1_29dof_dex1_joint_env_cfg import (
        PickPlaceG129DEX1JointWaistFixEnvCfg
    )
    
    # Extract arguments from args
    num_envs = getattr(args, 'num_envs', 1)
    device = getattr(args, 'device', 'cuda')
    env_spacing = getattr(args, 'env_spacing', 2.5)
    
    # Use the new factory
    env_cfg = PickPlaceG129DEX1JointWaistFixEnvCfg.create_with_params(
        num_envs=num_envs,
        device=device,
        env_spacing=env_spacing
    )
    
    # Set the task name
    env_cfg.env_name = args.task
    
    # Apply additional configuration
    apply_performance_optimizations(env_cfg, args)
    apply_termination_settings(env_cfg, args)
    
    return env_cfg


def apply_performance_optimizations(env_cfg, args: argparse.Namespace):
    """Apply performance optimization settings.
    
    Args:
        env_cfg: Environment configuration.
        args: Command-line arguments.
    """
    
    # Adjust simulation parameters for the precision mode
    precision_mode = getattr(args, 'precision_mode', 'balanced')
    
    if precision_mode == 'fast':
        # Fast mode: reduce precision to improve speed
        env_cfg.sim.dt = 0.01  # Larger time step
        if hasattr(env_cfg.sim, 'substeps'):
            env_cfg.sim.substeps = 1
            
    elif precision_mode == 'precise':
        # Accurate mode: improve precision, possibly at lower speed
        env_cfg.sim.dt = 0.002  # Smaller time step
        if hasattr(env_cfg.sim, 'substeps'):
            env_cfg.sim.substeps = 4
            
    else:  # balanced
        # Balanced mode: default settings
        env_cfg.sim.dt = 0.005
        if hasattr(env_cfg.sim, 'substeps'):
            env_cfg.sim.substeps = 2
    
    # Adjust the render rate based on optimization settings
    if getattr(args, 'disable_optimizations', False):
        # Use standard settings when optimization is disabled
        pass
    else:
        # Adjust settings when optimization is enabled
        if hasattr(env_cfg, 'decimation'):
            # Adjust decimation based on step_hz
            step_hz = getattr(args, 'step_hz', 500)
            if step_hz > 300:
                env_cfg.decimation = 2  # Higher control frequency
            elif step_hz < 100:
                env_cfg.decimation = 8  # Lower control frequency


def apply_termination_settings(env_cfg, args: argparse.Namespace):
    """Apply termination settings.
    
    Args:
        env_cfg: Environment configuration.
        args: Command-line arguments.
    """
    
    # Decide whether to remove timeout termination based on the options
    if getattr(args, 'disable_timeout', True):  # Disable timeout by default
        if hasattr(env_cfg.terminations, 'time_out'):
            env_cfg.terminations.time_out = None
    
    # Set the episode length
    episode_length = getattr(args, 'episode_length', 20.0)
    env_cfg.episode_length_s = episode_length


def add_env_config_args(parser: argparse.ArgumentParser):
    """Add environment configuration command-line arguments.
    
    Args:
        parser: Argument parser.
    """
    
    # Basic environment parameters
    env_group = parser.add_argument_group('Environment configuration')
    env_group.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    env_group.add_argument("--env_spacing", type=float, default=2.5, help="Environment spacing")
    env_group.add_argument("--episode_length", type=float, default=20.0, help="Episode length in seconds")
    env_group.add_argument("--disable_timeout", action="store_true", default=True, help="Disable timeout termination")
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation parameters')
    sim_group.add_argument("--sim_dt", type=float, default=0.005, help="Simulation time step")
    sim_group.add_argument("--substeps", type=int, default=2, help="Simulation substeps")
    
    # Performance optimization parameters
    perf_group = parser.add_argument_group('Performance optimization')
    perf_group.add_argument("--precision_mode", type=str, default="balanced", 
                           choices=["fast", "balanced", "precise"], help="Precision mode")
    perf_group.add_argument("--disable_optimizations", action="store_true", help="Disable all optimizations")


def print_env_config_info(env_cfg, args: argparse.Namespace):
    """Print environment configuration information.
    
    Args:
        env_cfg: Environment configuration.
        args: Command-line arguments.
    """
    
    print("\n=== Environment Configuration ===")
    print(f"Task name: {getattr(env_cfg, 'env_name', 'Unknown')}")
    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Environment spacing: {env_cfg.scene.env_spacing}")
    print(f"Replicate physics: {env_cfg.scene.replicate_physics}")
    print(f"Simulation time step: {env_cfg.sim.dt}")
    print(f"Episode length: {env_cfg.episode_length_s}s")
    print(f"Control frequency: {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.1f}Hz")
    print(f"Precision mode: {getattr(args, 'precision_mode', 'unknown')}")
    print(f"Optimizations: {'disabled' if getattr(args, 'disable_optimizations', False) else 'enabled'}")
    print("==================")


# Convenience helpers
def setup_env_from_args(args: argparse.Namespace):
    """Set environment configuration from command-line arguments.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        tuple: (env_cfg, env)
    """
    import gymnasium as gym
    
    # Create the environment configuration
    env_cfg = create_enhanced_env_cfg(args.task, args)
    
    # Print configuration information
    if getattr(args, 'verbose', False):
        print_env_config_info(env_cfg, args)
    
    # Create the environment
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    
    return env_cfg, env 