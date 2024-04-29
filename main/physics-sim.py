import numpy as np

class System2D():
    def __init__(self, m_ball=1, m_robot=.1, r_ball=1, h_robot=.1):
        self.m_ball = m_ball
        self.m_robot = m_robot
        self.r_ball = r_ball
        self.h_robot = h_robot
        self.theta_robot = 0
        self.theta_dot_robot = 0
        self.theta_ddot_robot = 0
        self.theta_dot_ball = 0
        self.theta_ddot = 0

    def reset_system_rand(self):
        # Resets System to chosen system state with certain amount of randomness
        # This method will be used to reset the system after a failed training loop
        # This method will reset these variables:
        # self.theta_robot = 0
        # self.theta_dot_robot = 0
        # self.theta_ddot_robot = 0
        # self.theta_dot_ball = 0
        # self.theta_ddot = 0
        pass

    def set_system(self):
        # Sets system to stated system state
        # This can be used to test trained system with various initial conditions
        # This method will set these variables to selected values:
        # self.theta_robot = 0
        # self.theta_dot_robot = 0
        # self.theta_ddot_robot = 0
        # self.theta_dot_ball = 0
        # self.theta_ddot = 0
        pass

    def step(self):
        # This method will step the system forward in time given the current system state
        pass

    def set_control(self):
        # This method will set the conrol variables given the current system state
        pass
