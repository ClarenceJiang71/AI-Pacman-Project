"""
Clarence Jiang
yunfan.jiang@emory.edu
THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY SOURCES
OUTSIDE OF THOSE APPROBED BY THE INSTRUCTOR. Clarence Jiang
"""

# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        """
        Test what each variable is like:
        print (successorGameState)
        print (newPos)
        print (currentFood)
        print (newFood)
        print (currentCapsules)
        print (newCapsules)
        print (newGhostStates)
        print (newScaredTimes)     
        """
        "*** YOUR CODE HERE ***"
        # print (successorGameState)
        # print (newPos)
        # print (currentFood)
        # print ("-" * 30)
        # print (newFood)
        # print (currentCapsules)
        # print (newCapsules)
        # print (newGhostStates[0].getPosition())
        # print (newScaredTimes)
        xpos, ypos = newPos
        food_list = currentFood.asList()
        food_distance_list = []
        """
        My idea is not to focus on capsule this time, as it is in a 
        fixed position at the very bottom right corner. The basic form
        of the evaluation function is to first look for the minimum distance
        with a food, second return the reciprocal of it. Two special cases
        are dealt with: 1. if min_distance is 0, return a large number to avoid
        dividing by 0. 2. if taking this action will run into a ghost, return 
        a really small number.        
        """
        for index in range(len(food_list)):
            food_distance = abs(xpos - food_list[index][0]) \
                                            + abs(ypos - food_list[index][1])
            food_distance_list.append(food_distance)
        min_food_distance = min(food_distance_list)

        if min_food_distance == 0:
            reciprocal = 1000
        else:
            reciprocal = 100/min_food_distance

        for ghost in newGhostStates:
            ghost_position = ghost.getPosition()
            if newPos == ghost_position:
                reciprocal = -1000000
        return reciprocal

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        """
        Requirements: 
        --see the tictactoe.py demo and the original AIMA code
            1.MinMaxPlayer method in tictactoe.py:
                def minimax_player(game, state):
                    'a minimax player'
                    if not game.terminal_test(state):
                        move = minimax_decision(state, game)
                        print "computer move: ", move
                        return move
                    else:
                        return
            2. minmax_decision method in tictactoe:
                def minimax_decision(state, game):
                player = game.to_move(state)
            
                def max_value(state):
                    if game.terminal_test(state):
                        return game.utility(state, player)
                    v = -infinity
                    for (a, s) in game.successors(state):
                        v = max(v, min_value(s))
                    return v
            
                def min_value(state):
                    if game.terminal_test(state):
                        return game.utility(state, player)
                    v = infinity
                    for (a, s) in game.successors(state):
                        v = min(v, max_value(s))
                    return v
            
                # Body of minimax_decision starts here:
                action, state = argmax(game.successors(state),
                                       lambda ((a, s)): min_value(s))
                return action
                
        --Code should work with any number number of ghosts, iterating agents. Minmax tree should have 
        multiple min layers for each ghost for every max layer.
        --Code should expand the game tree to a fixed depth.
        --Depth 2 search would involve pacman and every ghost moving two times. 
        
        --Picky on telling how many times to call GameState.getLegalActions

        """
        action_path = self.minimax_implementation(gameState, 0, 0)[1]
        return action_path


    def minimax_implementation(self, gameState, agent_counter, depth_level):

        # Retrieve number of agents which is important to check whether all of agents
        # take a move to next layer.
        num_of_agents = gameState.getNumAgents()

        #  A key update place where we need to move the agent pointer back to the
        #  start for the next layer. Each round of moving should start with the index-0
        #  agent which is the Pacman. Avoid agent_counter = 0 case, which representing the
        #  start of a round, so there is no need to update anything yet.
        if agent_counter % num_of_agents == 0 and agent_counter != 0:
            agent_counter = 0
            depth_level = depth_level + 1

        # The key variable to manipulate is the max_value and min value which will
        # keep track of all the changes of values of value associated with each node
        # during recursion
        if agent_counter == 0:
            value = float('-inf')
        else:
            value = float('inf')

        # This is the base case that our recursion tests on, if the depth exceeds our
        # depth limit, or if the node has no successor, what we return needs to track
        # two things, the current value of the node (which should be already adjusted
        # after min and max algorithm) and the action we take. Here, the action is
        # written as None, b/c those two cases indicate that we have already reached the
        # bottom, so there is no further action we should do. All we care about is the
        # value of those leaf nodes
        legal_actions = gameState.getLegalActions(agent_counter)
        if depth_level >= self.depth or not legal_actions:
            return self.evaluationFunction(gameState), None

        # Deal with Pacman and ghost separately, Pacman will always have index 0, so there
        # is no need to pass in the parameter of agent_counter, while ghost may have more
        # than one, so agent index is important for distinguishing.
        if agent_counter == 0:
            return self.max_action(legal_actions, gameState, value, depth_level)
        else:
            return self.min_action(legal_actions, gameState, value, agent_counter, depth_level)

    # max_action and min_action are simply comparing the actions and find the one action that
    # gives us the optimal value, either max or min depending on whether it's Pacman or ghost.
    def max_action(self, actions, gameState, max_value, depth_level):
        final_action = None
        for action in actions:
            successors = gameState.generateSuccessor(0, action)
            value = self.minimax_implementation(successors, 1, depth_level)[0]
            if value> max_value:
                max_value = value
                final_action = action
        return max_value, final_action

    def min_action(self, actions, gameState, min_value, agent_counter, depth_level):
        final_action = None
        for action in actions:
            successors = gameState.generateSuccessor(agent_counter, action)
            value = self.minimax_implementation(successors, agent_counter + 1, depth_level)[0]
            if value < min_value:
                min_value = value
                final_action = action
        return min_value, final_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-inf')
        beta = float('inf')
        action_path = self.minimax_implementation(gameState, 0, 0, alpha, beta)[1]
        return action_path

    def minimax_implementation(self, gameState, agent_counter, depth_level, alpha, beta):
        num_of_agents = gameState.getNumAgents()
        if agent_counter % num_of_agents == 0 and agent_counter != 0:
            agent_counter = 0
            depth_level = depth_level + 1

        #   This time we do not need to manipulate the value of negative infinity and infinity
        #   as the starting point anymore, since we will incorporate it within the max and min
        #   method. All the basic structure of this implementation method is the same.
        #   The differences are in the two sub-method.

        legal_actions = gameState.getLegalActions(agent_counter)
        if depth_level >= self.depth or not legal_actions:
            return self.evaluationFunction(gameState), None

        if agent_counter == 0:
            return self.max_action(legal_actions, gameState, depth_level, alpha, beta)
        else:
            return self.min_action(legal_actions, gameState, agent_counter, depth_level, alpha, beta)

    #   Basically my idea of max_action and min_action are just following the algorithm described
    #   in the pdf.
    def max_action(self, actions, gameState, depth_level, alpha, beta):
        final_action = None
        v = float('-inf')
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.minimax_implementation(successor, 1, depth_level, alpha, beta)[0]
            if value > v:
                v = value
                final_action = action
            if v > beta:
                return v, final_action
            if alpha < v:
                alpha = v
        return v, final_action

    def min_action(self, actions, gameState, agent_counter, depth_level, alpha, beta):
        final_action = None
        v = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(agent_counter, action)
            value = self.minimax_implementation(successor, agent_counter + 1, depth_level, alpha, beta)[0]
            if value < v:
                v = value
                final_action = action
            if v < alpha:
                return v, final_action
            if v < beta:
                beta = v
        return v, final_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

