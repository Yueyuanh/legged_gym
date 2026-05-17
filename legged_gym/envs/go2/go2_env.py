"""
Go2 environment with optional projectile support.

When cfg.projectile.enable=True, creates ProjectileManager during create_sim()
so that projectile actors exist before prepare_sim() and _init_buffers().
This avoids tensor-shape mismatches that would occur if projectiles were added
after env creation.
"""

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    torch_rand_float,
    quat_rotate_inverse,
    to_torch,
    get_axis_params,
)

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.projectile_manager import ProjectileManager


class Go2Env(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self._has_projectile = getattr(cfg, "projectile", None) and cfg.projectile.enable
        self._pm = None
        self._all_root_states = None
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def create_sim(self):
        super().create_sim()
        if self._has_projectile:
            self._pm = ProjectileManager(
                self.gym,
                self.sim,
                num_projectiles=self.cfg.projectile.num_projectiles,
                box_size=self.cfg.projectile.box_size,
                density=self.cfg.projectile.density,
            )

    def _init_buffers(self):
        if self._pm is None:
            super()._init_buffers()
            return

        # ---- acquire GPU state tensors (includes robot + projectile actors) ----
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # _all_root_states: full tensor for gym API calls.
        # root_states: robot-only view for all internal computations.
        self._all_root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_states = self._all_root_states[: self.num_envs]
        self.base_quat = self.root_states[:, 3:7]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # contact_forces: exclude projectile rigid bodies from the tail
        all_force = gymtorch.wrap_tensor(net_contact_forces)
        num_proj_bodies = self._pm.num_projectiles
        robot_force = all_force[: all_force.shape[0] - num_proj_bodies]
        self.contact_forces = robot_force.view(self.num_envs, -1, 3)

        # ---- initialize data buffers (same as LeggedRobot._init_buffers) ----
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
            device=self.device, requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device, requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
            device=self.device, requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs, len(self.feet_indices), dtype=torch.bool,
            device=self.device, requires_grad=False,
        )
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint position offsets and PD gains
        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device,
            requires_grad=False,
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                if self.cfg.control.control_type in ["P", "V"]:
                    print(
                        f"PD gain of joint {name} were not defined, setting them to zero"
                    )
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # ---- bind projectile manager tensors ----
        self._pm.bind_state_tensors(self._all_root_states, self.device)

    def _reset_root_states(self, env_ids):
        if self._pm is None:
            super()._reset_root_states(env_ids)
            return
        # Same logic as LeggedRobot._reset_root_states, but passes
        # _all_root_states to the gym API since it expects the full tensor.
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._all_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        if self._pm is None:
            super()._push_robots()
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self._all_root_states)
        )

    @property
    def projectile_manager(self):
        return self._pm
