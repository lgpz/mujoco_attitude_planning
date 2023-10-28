import os

import numpy as np
from spatialmath import SO3
import unittest

os.add_dll_directory("C://Users//Cybaster//.mujoco//mjpro150//bin")

from mujoco_py import load_model_from_path, MjSim, MjViewer

from src.motion_planning import *

from src.robot.robot import Robot


class TestThreeAttitude(unittest.TestCase):
    def test_three_attitude(self):
        model = load_model_from_path("../assets/universal_robots_ur5e/scene.xml")
        sim = MjSim(model)
        viewer = MjViewer(sim)

        dof = 6

        ur_robot = Robot()
        q0 = [0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0]
        ur_robot.set_joint(q0)
        T0 = ur_robot.get_cartesian()

        t0 = T0.t + np.array([0.1 * np.sqrt(2), -0.1 * np.sqrt(2), 0.0])
        R0 = SO3.Rz(-np.pi / 4) * SO3.Ry(-np.pi / 6) * SO3(T0.R)
        t1 = T0.t + np.array([0.1 * np.sqrt(2), 0.1 * np.sqrt(2), 0.0])
        R1 = SO3.Rz(np.pi / 4) * SO3.Ry(-np.pi / 3) * SO3(T0.R)

        tc = T0.t
        vec_n = np.array([0, 1, 1])

        arc_center_position_parameter = ArcCenterPositionParameter(t0, t1, tc)
        three_attitude_parameter = ThreeAttitudeParameter(R0, R1, vec_n)
        cartesian_parameter = CartesianParameter(arc_center_position_parameter, three_attitude_parameter)
        cubic_velocity_parameter = CubicVelocityParameter(10.0)
        trajectory_parameter = TrajectoryParameter(cartesian_parameter, cubic_velocity_parameter)

        trajectory_planner = TrajectoryPlanner(trajectory_parameter)

        dt = 0.02
        t_end = 20.0
        t_array = np.arange(0.0, t_end, dt)
        t_len = len(t_array)

        joints = np.zeros((t_len, dof))

        t_start = 5.0
        for i, ti in enumerate(t_array):
            Ti = trajectory_planner.interpolate(ti - t_start)
            ur_robot.move_cartesian(Ti)
            joints[i, :] = ur_robot.get_joint()

        t_step = 0
        forward = True
        j = 0

        while True:
            for i in range(dof):
                sim.data.qpos[i] = joints[t_step, i]
                sim.data.qvel[i] = 0.0

            sim.step()
            viewer.render()

            j += 1
            if j == 10:
                j = 0
                if forward:
                    t_step += 1
                    if t_step == t_len - 1:
                        forward = False
                else:
                    t_step -= 1
                    if t_step == 0:
                        forward = True
