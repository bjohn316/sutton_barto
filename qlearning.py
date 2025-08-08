import numpy as np
import random

class CustomEnvironment:
    def __init__(self, grid_size=4):
        """
        Custom grid environment.
        
        Parameters:
            grid_size: Size of the grid (grid_size x grid_size).
        """
        self.grid_size = grid_size
        self.state_space = grid_size * grid_size
        self.action_space = 4  # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.reset()

    def reset(self):
        """Resets the environment to the starting state."""
        self.agent_position = [0, 0]  # Start at top-left corner
        return self._get_state()

    def step(self, action):
        """
        Takes an action and returns the next state, reward, done, and info.
        
        Parameters:
            action: The action to take (0=Up, 1=Right, 2=Down, 3=Left).
        
        Returns:
            next_state: The next state after taking the action.
            reward: Reward for the action.
            done: Whether the episode is finished.
            info: Additional information (empty dictionary).
        """
        if action == 0:  # Up
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 1:  # Right
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)
        elif action == 2:  # Down
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)
        elif action == 3:  # Left
            self.agent_position[1] = max(0, self.agent_position[1] - 1)

        next_state = self._get_state()
        reward = 1 if self.agent_position == [self.grid_size - 1, self.grid_size - 1] else 0
        done = self.agent_position == [self.grid_size - 1, self.grid_size - 1]
        return next_state, reward, done, {}

    def _get_state(self):
        """Converts the agent's position to a single integer state."""
        return self.agent_position[0] * self.grid_size + self.agent_position[1]

# Q-Learning implementation
def q_learning(env, num_episodes, alpha, gamma, epsilon):
    """
    Q-Learning algorithm based on Sutton and Barto's Reinforcement Learning book.

    Parameters:
        env: The custom environment.
        num_episodes: Number of episodes to train.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate for epsilon-greedy policy.

    Returns:
        Q: The Q-table containing state-action values.
    """
    # Initialize Q-table with zeros
    Q = np.zeros((env.state_space, env.action_space))

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, env.action_space - 1)  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit

            # Take action and observe reward and next state
            next_state, reward, done, _ = env.step(action)

            # Update Q-value using the Q-Learning update rule
            best_next_action = np.argmax(Q[next_state, :])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # Transition to next state
            state = next_state

    return Q

# Example usage
if __name__ == "__main__":
    # Create custom environment
    env = CustomEnvironment(grid_size=4)

    # Hyperparameters
    num_episodes = 1000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    # Train Q-Learning agent
    Q = q_learning(env, num_episodes, alpha, gamma, epsilon)

    print("Trained Q-Table:")
    print(Q)