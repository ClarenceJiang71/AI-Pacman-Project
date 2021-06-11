"""
Clarence Jiang
yunfan.jiang@emory.edu
THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY SOURCES
OUTSIDE OF THOSE APPROBED BY THE INSTRUCTOR. Clarence Jiang
"""



# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # First start with the outermost iteration should be going through number of iterations
        for i in range(iterations):
            # Some initial setup include the states information and the temp_value Counter
            # It is important to keep a copy of Counter; updating within any for-loop will
            # eventually lead to an error, so we have to use an extra data structure
            states = self.mdp.getStates()
            temp_value = self.values.copy()
            # Check through each state, remember that one state + one action is called a Q.
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if actions: # Check if the action is empty, if it is, go to else
                    value_list = [] # Use a value_list to store the Q_value for each action
                    for action in actions:
                        Q_value = self.computeQValueFromValues(state, action)
                        value_list.append(Q_value)
                    temp_value[state] = max(value_list) # Assign the max as the value of this state
                else:
                    #   The state without any subsequent action, just assign 0 value to it similar to
                    #   a leaf node
                    temp_value[state] = 0
            self.values = temp_value


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        """
        My implementation idea is simply based on the 4th Markov Decision Process Video 
        which introduces the basic formula of computing Q value from values. 
        """
        Q = 0
        #   Each component of the successive values returned is a tuple that contains the
        #   next state and the probability which is the weight.
        successive_values = self.mdp.getTransitionStatesAndProbs(state, action)
        for index in range(len(successive_values)):
            T_weight = successive_values[index][1]
            reward_Q = self.mdp.getReward(state, action, successive_values[index][0]) \
                 + self.discount * self.values[successive_values[index][0]]
            Q += reward_Q * T_weight
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        """
        This is simply applying our method of Q-value calculation right defined above.
        Since we have access to the Q value for each action, we can just simply iterate 
        and always keep track of the one action that results into the highest Q value. 
        I simply just use two pointer to keep track of value and action respectively, 
        and eventually output the action. 
        
        I realize it's sort of hard to use a similar implementation as the one I used to 
        calculate the Q value, since it's not easy to trace back which action exactly corresponds
        to the optimal Q value. 
        """
        if not self.mdp.getPossibleActions(state):
            return None
        else:
            actions = self.mdp.getPossibleActions(state)
            #   Introduce two pointers
            final_action_pointer = None
            final_action_value = -100000
            #   Iterate through each action to see which action results into the optimal
            for action in actions:
                Q = self.computeQValueFromValues(state, action)
                if Q >= final_action_value:
                    final_action_pointer = action
                    final_action_value = Q
            return final_action_pointer

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
