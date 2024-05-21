import numpy as np
from scipy.linalg import expm

class System2D():
    def __init__(self, dt=.01):
        """
        Initializes a 2D system with default or provided parameters.
        """

        self.m_B = 51.66 # mass of robot body (kg)
        self.m_b = 2.44 # mass of ball (kg)
        self.l_com = 0.69 # distance from ball COM to total COM (m)
        self.I_b = 0.0174 # ball moment of inertia (kg-m^2)
        self.r_b = 0.1058 # ball radius (m)
        self.I_B = 12.59 # intertia of body wrt center of ball (kg-m^2)
        self.mu_theta = 3.68 # theta viscous damping coefficient (N-m-s/rad)
        self.mu_phi = 3.68 # phi viscous damping coefficient (N-m-s/rad)
        self.dt = dt
        self.g = 9.81 #gravitational constant (m/s^2)

        self.theta = 0 # ball angular configuration
        self.phi = 0 # body angle
        self.theta_dot = 0 # change in ball angular configuration
        self.phi_dot = 0 # change in body angle

        # Larger calculated values to simplify dynamic matrices
        self.gamma_1 = self.I_B + self.I_b + self.m_b * self.r_b ** 2 + self.m_B * self.r_b ** 2 + self.m_B * self.l_com
        self.gamma_2 = self.I_B + self.m_B * self.l_com ** 2
        self.F = self.gamma_2 + self.m_B * self.r_b * self.l_com
        self.G = self.gamma_1 + 2 * self.m_B * self.r_b * self.l_com
        self.H = self.gamma_2 ** 2 - self.gamma_1 * self.gamma_2 + (self.m_B * self.l_com * self.r_b) ** 2

        # Continuous time dynamics
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [self.m_B**2 * self.l_com**2 * self.g * self.r_b / self.H, self.m_B**2 * self.l_com**2 * self.g * self.r_b / self.H, self.mu_theta * self.gamma_2 / self.H, -self.mu_phi * self.F / self.H],
            [self.m_B * self.g * self.l_com * (self.F - self.G) / self.H, self.m_B * self.g * self.l_com * (self.F - self.G) / self.H, -self.mu_theta * (self.gamma_2 + self.m_B * self.r_b * self.l_com) / self.H, self.mu_phi * self.G / self.H]
        ])

        self.B = np.array([
            [0],
            [0],
            [self.F / self.H],
            [-self.G / self.H]
        ])
        
        # Discretized dynamics
        self.A_dynamics = expm(self.A * self.dt)
        self.B_dynamics = np.linalg.inv(self.A) @ (self.A_dynamics - np.eye(self.A.shape[0])) @ self.B

         

    def reset_system_rand(self):
        """
        Resets the system to a random state.

        This method resets the following variables:
        - self.theta
        - self.phi
        - self.theta_dot
        - self.phi_dot
        
        This method is typically used after a failed training loop to restart with new conditions.
        """

        self.theta = np.random.uniform(-np.pi, np.pi)
        self.phi = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = np.random.uniform(-1, 1)
        self.phi_dot = np.random.uniform(-1, 1)


    def set_system(self, theta, phi, theta_dot, phi_dot):
        """
        Sets the system to a specified state.

        This method sets the following variables to specific values:
        - self.theta
        - self.phi
        - self.theta_dot
        - self.phi_dot

        This can be used for testing the system with different initial conditions.
        """
        self.theta = theta
        self.phi = phi
        self.theta_dot = theta_dot
        self.phi_dot = phi_dot



    def step(self, u):
        """
        Steps the system forward in time based on the current state.

        This method calculates the next state of the system based on its current state
        and any control inputs.
        """
        x = np.array([self.theta, self.phi, self.theta_dot, self.phi_dot])
        x_next = self.A_dynamics @ x + self.B_dynamics @ u

        self.theta, self.phi, self.theta_dot, self.phi_dot = x_next


    
    # This is something that you'll probably have to set up using the ML algorithms, Cameron. For now,
    # I have it set up using a basic feedback control law to control the torque, where our input u, the torque
    # is defined by u = -Kx, where K is the feedback gain vector and x are our state variables
    def set_control(self, K):
        """
        Sets the control variables based on the current system state.

        This method calculates the control input u based on the current state using
        a state feedback control law u = -Kx.
        """
        x = np.array([self.theta, self.phi, self.theta_dot, self.phi_dot])
        u = -K @ x
        return u

    def get_observation_space(self):
        """
        Returns the observation space of the system as a numpy array.

        The observation space includes the following parameters:
        - theta_robot: Angle of the robot.
        - theta_dot_robot: Angular velocity of the robot.
        - theta_dot_ball: Angular velocity of the ball.
        - theta_ddot_ball: Angular acceleration of the ball.

        Returns:
            numpy.ndarray: A numpy array containing the observation space parameters 
                        [theta_robot, theta_dot_robot, theta_ddot_robot, theta_dot_ball, theta_ddot_ball].
        
        Example:
            >>> obj = YourClass()
            >>> observation_space = obj.get_observation_space()
            >>> print(observation_space)
            [theta_robot_value, theta_dot_robot_value, theta_ddot_robot_value, theta_dot_ball_value, theta_ddot_ball_value]
        """
        return np.array[self.theta_robot, self.theta_dot_robot,
                        self.theta_dot_ball, self.theta_ddot_ball]
    
    def get_action_space(self):
        """
        Returns the action space of the system as a numpy array.

        The action space is defined as the range of possible actions that can be 
        taken by the system. In this case, it is a continuous range between -10 and 10.

        Returns:
            numpy.ndarray: A numpy array containing the lower and upper bounds of 
                        the action space, i.e., [-10, 10].
        
        Example:
            >>> obj = YourClass()
            >>> action_space = obj.get_action_space()
            >>> print(action_space)
            [-10, 10]
        """
        return np.array([-10, 10])
