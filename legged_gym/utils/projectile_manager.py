"""
Projectile Manager for LeggedGym
Handles spawning and managing projectiles for interactive training
"""

import numpy as np
from isaacgym import gymapi, gymtorch
import torch

class ProjectileManager:
    def __init__(self, gym, sim, num_projectiles=20, box_size=0.3, density=10.0):
        """
        Initialize projectile manager
        
        Args:
            gym: Isaac Gym instance
            sim: simulation instance
            num_projectiles: number of projectiles to create (for cycling)
            box_size: size of projectile box
            density: density of projectile material
        """
        self.gym = gym
        self.sim = sim
        self.num_projectiles = num_projectiles
        self.projectiles = []
        self.projectile_actor_indices = []
        self.proj_index = 0
        self.proj_env = None
        self.box_size = box_size
        self.density = density
        # Keep idle projectiles far from the default viewer frustum.
        self.hidden_base_pos = (-80.0, -80.0, -20.0)
        self.hidden_spacing = 0.5
        self.root_states = None
        self._indices_int32 = None
        self._indices_long = None
        
        # Create a separate environment for projectiles
        self._create_projectile_environment()

    def _hidden_position(self, i):
        return (
            self.hidden_base_pos[0] + i * self.hidden_spacing,
            self.hidden_base_pos[1],
            self.hidden_base_pos[2],
        )
        
    def _create_projectile_environment(self):
        """Create a separate environment to host projectiles"""
        # Create a simple environment bounds
        lower = gymapi.Vec3(-100.0, -100.0, -100.0)
        upper = gymapi.Vec3(100.0, 100.0, 100.0)
        self.proj_env = self.gym.create_env(self.sim, lower, upper, 1)
        
        # Create projectile asset
        proj_asset_options = gymapi.AssetOptions()
        proj_asset_options.density = self.density
        proj_asset = self.gym.create_box(
            self.sim, 
            self.box_size, 
            self.box_size, 
            self.box_size, 
            proj_asset_options
        )
        
        # Create projectiles
        for i in range(self.num_projectiles):
            pose = gymapi.Transform()
            hx, hy, hz = self._hidden_position(i)
            pose.p = gymapi.Vec3(hx, hy, hz)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            
            # Create actor that will collide with any environment
            ahandle = self.gym.create_actor(
                self.proj_env, 
                proj_asset, 
                pose, 
                f"projectile_{i}", 
                -1,  # -1 means collide with all environments
                0
            )
            
            # Set random color for each projectile
            c = 0.5 + 0.5 * np.random.random(3)
            self.gym.set_rigid_body_color(
                self.proj_env, 
                ahandle, 
                0, 
                gymapi.MESH_VISUAL_AND_COLLISION, 
                gymapi.Vec3(c[0], c[1], c[2])
            )
            
            self.projectiles.append(ahandle)
            actor_idx = self.gym.get_actor_index(self.proj_env, ahandle, gymapi.DOMAIN_SIM)
            self.projectile_actor_indices.append(actor_idx)

    def bind_state_tensors(self, all_root_states, device):
        """Bind to global actor root-state tensor managed by the environment."""
        self.root_states = all_root_states
        if len(self.projectile_actor_indices) == 0:
            return
        self._indices_int32 = torch.tensor(self.projectile_actor_indices, dtype=torch.int32, device=device)
        self._indices_long = self._indices_int32.to(dtype=torch.long)
        self.reset_all_projectiles()
    
    def fire_projectile_from_camera(self, viewer, speed=25.0, add_random_spin=True):
        """
        Fire a projectile from current camera position and orientation
        
        Args:
            viewer: gym viewer instance
            speed: initial speed of projectile
            add_random_spin: whether to add random angular velocity
        
        Returns:
            bool: True if projectile was fired successfully
        """
        if not viewer or not self.proj_env:
            return False
        
        # Get camera transform
        cam_pose = self.gym.get_viewer_camera_transform(viewer, self.proj_env)
        cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1))  # Camera forward direction
        
        # Calculate spawn position and velocity
        spawn = cam_pose.p
        vel = cam_fwd * speed
        
        # Optional random angular velocity
        angvel = None
        if add_random_spin:
            angvel = 1.57 - 3.14 * np.random.random(3)
        
        # Fire the projectile
        return self.fire_projectile(spawn, vel, angvel)
    
    def fire_projectile(self, position, linear_velocity, angular_velocity=None):
        """
        Fire a projectile from specified position with given velocity
        
        Args:
            position: gymapi.Vec3 or list/tuple of 3 floats for spawn position
            linear_velocity: gymapi.Vec3 or list/tuple of 3 floats for initial velocity
            angular_velocity: optional gymapi.Vec3 or list/tuple of 3 floats for angular velocity
        
        Returns:
            bool: True if projectile was fired successfully
        """
        if self.root_states is None or self._indices_int32 is None:
            return False

        # Convert inputs to gymapi.Vec3 if needed
        if not isinstance(position, gymapi.Vec3):
            position = gymapi.Vec3(position[0], position[1], position[2])
        if not isinstance(linear_velocity, gymapi.Vec3):
            linear_velocity = gymapi.Vec3(linear_velocity[0], linear_velocity[1], linear_velocity[2])

        # Get next projectile to use (cycling)
        actor_idx = self._indices_long[self.proj_index]

        # Update actor root state through tensor API (GPU-pipeline safe)
        self.root_states[actor_idx, 0:3] = torch.tensor(
            [position.x, position.y, position.z], device=self.root_states.device, dtype=self.root_states.dtype
        )
        self.root_states[actor_idx, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.root_states.device, dtype=self.root_states.dtype
        )
        self.root_states[actor_idx, 7:10] = torch.tensor(
            [linear_velocity.x, linear_velocity.y, linear_velocity.z], device=self.root_states.device, dtype=self.root_states.dtype
        )

        if angular_velocity is not None:
            if not isinstance(angular_velocity, gymapi.Vec3):
                angular_velocity = gymapi.Vec3(angular_velocity[0], angular_velocity[1], angular_velocity[2])
            self.root_states[actor_idx, 10:13] = torch.tensor(
                [angular_velocity.x, angular_velocity.y, angular_velocity.z], device=self.root_states.device, dtype=self.root_states.dtype
            )
        else:
            self.root_states[actor_idx, 10:13] = 0.0

        actor_idx_int32 = self._indices_int32[self.proj_index:self.proj_index + 1]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_idx_int32),
            1
        )
        
        # Update index for next projectile
        self.proj_index = (self.proj_index + 1) % len(self.projectiles)
        
        return True
    
    def reset_all_projectiles(self):
        """Reset all projectiles to initial positions (far away)"""
        if self.root_states is None or self._indices_int32 is None:
            return

        for i in range(len(self.projectile_actor_indices)):
            actor_idx = self._indices_long[i]
            hx, hy, hz = self._hidden_position(i)
            self.root_states[actor_idx, 0:3] = torch.tensor(
                [hx, hy, hz], device=self.root_states.device, dtype=self.root_states.dtype
            )
            self.root_states[actor_idx, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.root_states.device, dtype=self.root_states.dtype
            )
            self.root_states[actor_idx, 7:13] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._indices_int32),
            len(self.projectile_actor_indices)
        )
        self.proj_index = 0
    
    def get_projectile_positions(self):
        """
        Get current positions of all projectiles
        
        Returns:
            list of gymapi.Vec3 positions
        """
        if self.root_states is None or self._indices_long is None:
            return []

        self.gym.refresh_actor_root_state_tensor(self.sim)
        positions = []
        pos_tensor = self.root_states[self._indices_long, 0:3]
        for i in range(pos_tensor.shape[0]):
            pos = pos_tensor[i]
            positions.append(gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2])))
        return positions
    
    def get_projectile_positions_tensor(self, device='cuda:0'):
        """
        Get current positions of all projectiles as a torch tensor
        
        Args:
            device: torch device to place tensor on
            
        Returns:
            torch.Tensor of shape (num_projectiles, 3)
        """
        if self.root_states is None or self._indices_long is None:
            return torch.empty((0, 3), device=device, dtype=torch.float32)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        return self.root_states[self._indices_long, 0:3].to(device=device, dtype=torch.float32)
