import numpy as np

class System2D():
    def __init__(self, m_ball=1, m_robot=.1, r_ball=1, h_robot=.1, dt=.01):
        """
        Initializes a 2D system with default or provided parameters.

        Parameters:
        - m_ball (float): Mass of the ball. (kg)
        - m_robot (float): Mass of the robot. (kg)
        - r_ball (float): Radius of the ball. (m)
        - h_robot (float): Height of the robot. (m)
        - dt (float): Time step size (s)
        """
        self.m_ball = m_ball
        self.m_robot = m_robot
        self.r_ball = r_ball
        self.h_robot = h_robot
        self.dt = dt
        self.theta_robot = 0
        self.theta_dot_robot = 0
        self.theta_ddot_robot = 0
        self.theta_dot_ball = 0
        self.theta_ddot = 0

    def reset_system_rand(self):
        """
        Resets the system to a random state.

        This method resets the following variables:
        - self.theta_robot
        - self.theta_dot_robot
        - self.theta_ddot_robot
        - self.theta_dot_ball
        - self.theta_ddot

        This method is typically used after a failed training loop to restart with new conditions.
        """
        pass

    def set_system(self):
        """
        Sets the system to a specified state.

        This method sets the following variables to specific values:
        - self.theta_robot
        - self.theta_dot_robot
        - self.theta_ddot_robot
        - self.theta_dot_ball
        - self.theta_ddot

        This can be used for testing the system with different initial conditions.
        """
        pass

    def step(self):
        """
        Steps the system forward in time based on the current state.

        This method calculates the next state of the system based on its current state
        and any control inputs (not yet implemented).
        """
        pass

    def set_control(self):
        """
        Sets the control variables based on the current system state.

        This method sets the following variables to specific values:
        - self.theta_dot_robot
        - self.theta_ddot_robot

        This method will implement logic to set control inputs for the system based on
        its current state. This is used to update control strategies during simulation.
        """
        pass
