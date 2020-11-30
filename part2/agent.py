############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import time
import random

import matplotlib.pyplot as plt

from collections import deque 

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 450
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        self.episode_number =0
        #define DQN object
        self.dqn = DQN()

        #Epsilon burst determines whether we want to increase the epsilon for the second part of an episode to do more exploring
        self.epsilon_burst = False
        self.epsilon = 1
        self.late_state_epsilon = 0.9
        self.edr = 0.98

        #make reward a class variable 
        self.distance_to_goal = 1

        #target network update rate
        self.target_update_rate = 20
        self.time = time.time()

        self.losses = []
        self.mean_losses = []

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        #if episode has ended 
        if self.num_steps_taken % self.episode_length == 0:
            #diff = time.time() - self.time 
            #self.time = time.time()
            self.episode_number = self.num_steps_taken/self.episode_length
            #print(("e: {:.2f} lse: {:.2f} ep: {} d: {} t = {:.2f}").format(self.epsilon,self.late_state_epsilon,self.episode_number,self.distance_to_goal,diff))
            
            # self.mean_losses.append(np.mean(self.losses))
            # self.losses = []
            
            #self.epsilon = min(1,(self.epsilon + (1/max(1,episode_number))*(self.distance_to_goal - self.epsilon)))
            self.epsilon = max(0.1,self.epsilon*self.edr)
            # if self.episode_number == 70:
            #     self.dqn.replay_buffer.prioritise = True
            return True
        else:
            return False

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move left
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        elif discrete_action == 2:
            # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32) 
        else:
            # Move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        
        
        return continuous_action
    
    # Function for the agent to choose its next action
    def _choose_next_action(self):

        #the best next action is the one which gives us the best reward
        q_values = self.dqn.q_network.forward(torch.tensor(self.state))
        max_q_index = q_values.max(0)[1] 
       
        #if we are more than 60% into the episode and over 25 episodes, make epsilon slightly higher           
        if self.epsilon_burst and self.num_steps_taken - (self.episode_length*self.episode_number) > self.episode_length*0.6 and self.episode_number > 25:
            #make the agent explore more in the second half of the episode 
            self.late_state_epsilon= self.late_state_epsilon*0.9995
             #make a probability distribution 
            epsilon_pd = [self.late_state_epsilon/4,self.late_state_epsilon/4,self.late_state_epsilon/4,self.late_state_epsilon/4]
            epsilon_pd[max_q_index] = 1 - self.late_state_epsilon + self.late_state_epsilon/4       
            #print("epsilon change to:",epsilon)
        else:
            #make a probability distribution 
            epsilon_pd = [self.epsilon/4,self.epsilon/4,self.epsilon/4,self.epsilon/4]
            epsilon_pd[max_q_index] = 1 - self.epsilon + self.epsilon/4

        #choose the actual action based on epsilon
        action = random.choices([0,1,2,3],weights=epsilon_pd)

        return action[0]

    
    def get_next_action(self, state):
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state

        #get a discrete actiion
        action = self._choose_next_action()
        # Store the action; this will be used later, when storing the transition
        self.action = action
        
        return self._discrete_action_to_continuous(action)

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        self.distance_to_goal = distance_to_goal
        
        #assign a slightly lower reward to the wall
        if self.state.tolist() != next_state.tolist():
            reward = 1 - self.distance_to_goal
        else:
            reward = 0.8*(1 - self.distance_to_goal)

    #update target network 
        if self.num_steps_taken%self.target_update_rate == 0:
            self.dqn.update_target_network()
        
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        
        #add transition to replay buffer
        self.dqn.replay_buffer.add_transition(transition)
        
        #once the buffer has enough transitinos in it, we can start to sample the buffer to create a minibatch to train with 
        if len(self.dqn.replay_buffer.buffer) > self.dqn.replay_buffer.minibatch_length:
            
            minibatch = self.dqn.replay_buffer.get_minibatch()
            #calculate loss 
            loss = self.dqn.train_q_network(minibatch)
            #self.losses.append(loss)

        #update probabilities in the prioritised replay buffer
        self.dqn.replay_buffer.update_p()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        action = self.dqn.q_network.forward(torch.tensor(state)).argmax(0)

        return self._discrete_action_to_continuous(action)

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

        #define replay buffer object
        self.replay_buffer = ReplayBuffer()

        #set if yolu want double q learning or not 
        self.DoubleQ = True 

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

        #two cases depending on if double Q learning is enabled 
        if not self.DoubleQ:       
            #predict the 4 rewards for differnt directions from the next state 
            network_prediction_NS = self.target_network.forward(next_state_tensor)
            max_predicted_q_values = network_prediction_NS.max(1)[0].detach()
        else:
  
            #get prediction
            network_prediction_NS = self.q_network.forward(next_state_tensor)
            
            #get max q value index based on the target network 
            max_q_value_indices = self.target_network.forward(next_state_tensor).detach().argmax(1)
            max_predicted_q_values = network_prediction_NS.gather(1,max_q_value_indices.unsqueeze(1)).squeeze(1)

        actual_return = reward_tensor + 0.9*max_predicted_q_values

        #get predicted reward in the chosen direction
        predicted_reward = network_prediction_R.gather(1,action_tensor.unsqueeze(1)).squeeze(1)

        err = abs(predicted_reward - actual_return).detach().numpy()
        self.replay_buffer.update_weights(err)

        loss = torch.nn.MSELoss()(predicted_reward,actual_return.float())

        return loss 
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class ReplayBuffer():
    

    def __init__(self ):
        
        self.buffer = deque(maxlen=7000)
        self.minibatch_length = 150
        self.minibatch_indices = []
        self.weights = deque(maxlen=7000)
        self.p = []
        self.alpha = 0.7
        self.probability_constant = 0.001
        self.prioritise =  True
        self.count = 0
    def add_transition(self,transition):
        self.buffer.append(transition)
        #add the max weights to the list, or add 1 if its empty 
        self.weights.append(max(self.weights)) if len(self.weights) > 0 else self.weights.append(1)
    
    def get_minibatch(self):

        minibatch = []
        #if prioritised replay is set to true 
        if self.prioritise:
            #tackling edge cases for sampling indices 
            if len(self.buffer) <=7000 and self.count == 0:
                indices_range = range(len(self.buffer)-1) 
                if(len(self.buffer)) == 7000:
                    self.count += 1
            else:
                indices_range = range(7000)

            #choose random indices 
            self.minibatch_indices = random.choices(indices_range,weights = self.p,k = self.minibatch_length)    

            #make the minibatch 
            for i in self.minibatch_indices:
                minibatch.append(self.buffer[i])

        else:
            minibatch = random.sample(self.buffer,self.minibatch_length)
        return minibatch


    def update_weights(self,errors):
        self.probability_constant = 0.05*max(errors)
        for x,i in enumerate(self.minibatch_indices):
            self.weights[i] = errors[x] + self.probability_constant


    def update_p(self):
        #make p an empty list of the same length as weights 
        if self.prioritise:
            self.p = np.zeros(len(self.weights))
            sum_weights = sum(np.array(self.weights)**self.alpha)
            self.p = (np.array(self.weights)**self.alpha)/sum_weights
        else: 
            self.p = np.zeros(len(self.weights))