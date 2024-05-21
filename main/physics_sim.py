import numpy as np
import jax
import jax.numpy as jnp

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

    def set_system(self, theta_robot, theta_dot_robot, theta_ddot_robot, theta_dot_ball, theta_ddot):
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
        self.theta_robot = theta_robot
        self.theta_dot_robot = theta_dot_robot 
        self.theta_ddot_robot = theta_ddot_robot
        self.theta_dot_ball = theta_dot_ball
        self.theta_ddot_robot = theta_ddot

    def ode(self):
        """
        Continuous time dynamics of ball balancing robot
        """
        pass
        
    def discrete_step(self, state, control, dt):
        """
        Steps the system forward in time based on the current state.

        This method calculates the next state of the system based on its current state
        and any control inputs (not yet implemented).
        """
        return state + dt * self.ode(state, control)

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
