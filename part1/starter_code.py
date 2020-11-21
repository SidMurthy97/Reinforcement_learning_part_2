# Import some modules from other libraries
import numpy as np
import torch
import time
import random
from tqdm import trange
import matplotlib.pyplot as plt
# Import the environment module
from environment import Environment
from q_value_visualiser import QValueVisualiser
from collections import deque 
import pprint

# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

        #epsilon
        self.epsilon = 1

        #epsilon decay rate
        self.edr = 0.9999

        #make an instance of the DQN class
        self.dqn = DQN()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return discrete action 0
        #the best next action is the one which gives us the best reward
        q_values = self.dqn.q_network.forward(torch.tensor(self.state))
        max_q_index = q_values.max(0)[1]

        #we need to now give it some epsilon randomness
        self.epsilon = self.epsilon*self.edr
        
        #make a probability distribution 
        epsilon_pd = [self.epsilon/4,self.epsilon/4,self.epsilon/4,self.epsilon/4]
        epsilon_pd[max_q_index] = 1 - self.epsilon + self.epsilon/4


        action = random.choices([0,1,2,3],weights=epsilon_pd)
        #action = random.randint(0,3)
        return action[0]

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move right
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move left
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 2:
            # Move up
            continuous_action = np.array([0, 0.1], dtype=np.float32) 
        else:
            # Move down
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        
        
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.target_network = Network(input_dimension=2, output_dimension=4)
        
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch):
        
        #split minibatch into 4 arrays
        state,action,reward,next_state = zip(*minibatch)

        #make tensors 
        state_tensor = torch.tensor(state)
        action_tensor = torch.tensor(action)
        reward_tensor = torch.tensor(reward)
        next_state_tensor = torch.tensor(next_state)
        
        #get predicted rewards for the 4 directions 
        network_prediction_R = self.q_network.forward(state_tensor)
        
        #predict the 4 rewards for differnt directions from the next state 
        network_prediction_NS = self.target_network.forward(next_state_tensor)
        max_predicted_q_values = network_prediction_NS.max(1)[0].detach()

        actual_return = reward_tensor + 0.9*max_predicted_q_values

        #get predicted reward in the chosen direction
        predicted_reward = network_prediction_R.gather(1,action_tensor.unsqueeze(1)).squeeze(1)

        loss = torch.nn.MSELoss()(predicted_reward,actual_return)

        return loss 
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_optimal_policy(self):
        optimal_q_values = np.zeros([10,10])
        all_q_values = np.zeros([10,10,4])
        for x,i in enumerate(np.arange(0.05,1,0.1)):
            
            for y,j in enumerate(np.arange(0.05,1,0.1)):
                y = 9-y  
                state = np.array([i,j], dtype=np.float32)

                network_prediction = self.q_network.forward(torch.tensor(state))
                all_q_values[y][x] = network_prediction.detach().numpy()
                #print(all_q_values,all_q_values.max(0)[1])
                optimal_q_values[y][x] = network_prediction.max(0)[1]

        return optimal_q_values,all_q_values

class ReplayBuffer():
    
    def __init__(self):
        self.buffer = deque(maxlen=5000)

    def add_transition(self,transition):
        self.buffer.append(transition)
    
    def get_minibatch(self,minibatch_length):
        minibatch = random.sample(self.buffer,minibatch_length)
        return minibatch

# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    
    replay_buffer = ReplayBuffer()
    n_episodes = 100
    n_steps = 100
    minibatch_length = 100
    target_network_update_rate = 1
    losses = []

    loss_per_episode = []
    # Loop over episodes
    for episode in trange(n_episodes):
        # Reset the environment for the start of the episode.
        agent.reset()

        #update target network 
        if episode % target_network_update_rate == 0:
            agent.dqn.update_target_network()
            

        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(n_steps):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            
            #add transition to replay buffer
            replay_buffer.add_transition(transition)

            #once the buffer has enough transitinos in it, we can start to sample the buffer to create a minibatch to train with 
            if len(replay_buffer.buffer) > minibatch_length:
                
                minibatch = replay_buffer.get_minibatch(minibatch_length)
                #calculate loss 
                loss = agent.dqn.train_q_network(minibatch)
                losses.append(loss)
                # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
                #time.sleep(0.2)
        
        #only take mean when losses are being recoreed
        if len(losses) > 0:
            loss_per_episode.append(np.mean(losses))
            losses = []
    
    #now that training is complete, we need to get the optimal policy
    policy,q_values = agent.dqn.get_optimal_policy()
    
    environment.draw_policy(policy,100)
    visualiser = QValueVisualiser(environment=environment, magnification=500)
    visualiser.draw_q_values(q_values)
    pprint.pprint(policy)
    plt.plot(loss_per_episode)
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Episodes')
    plt.title(('Minibatch size = {}, steps per episode = {}').format(minibatch_length,n_steps))
    plt.show()