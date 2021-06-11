"""
Clarence Jiang
yunfan.jiang@emory.edu
THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING ANY SOURCES
OUTSIDE OF THOSE APPROBED BY THE INSTRUCTOR. Clarence Jiang
"""

# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    """
    test area collection-
    # print "Start:", problem.getStartState()
    # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    # print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    start_state = problem.getStartState()   # the start state of the problem
    visited = []    # the states that already visited
    node_stack = []  # the stack for storing nodes, we need to test if the node is the goal state
    path_stack = []  # the stack for path, we not only need the state, but also the path to the goal
    path = []   # this is what we will return

    # test if the start state is simply a goal state, if it is, we just return an empty path
    if problem.isGoalState(start_state):
        return path

    # Before start, put the start state into the node stack
    node_stack.append(start_state)

    while True:
        state = node_stack.pop()  # pop out the state currently working on
        if len(path_stack) != 0:    # prepare for future pop of path
            path = path_stack.pop()

        if problem.isGoalState(state):  # if reach a goal state, end this while-loop
            return path

        visited.append(state)  # add the state that has already expanded
        successors = problem.getSuccessors(state)    # get the list of successors

        if len(successors) != 0:
            for node in successors:
                if node[0] in visited:
                    continue
                else:
                    node_stack.append(node[0])
                    # it is vital to keep the concatenated form, or part of the path might be lose
                    # or the size of expanded nodes tend to be large
                    concatenated = path + [node[1]]
                    path_stack.append(concatenated)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    start_state = problem.getStartState()  # the start state of the problem
    visited = []  # the states that already visited
    node_queue = []  # the queue for storing nodes, we need to test if the node is the goal state
    path_queue = []  # the queue for path, we not only need the state, but also the path to the goal
    path = []  # this is what we will return

    # test if the start state is simply a goal state, if it is, we just return an empty path
    if problem.isGoalState(start_state):
        return path

    # Before start, put the start state into the node queue
    node_queue.append(start_state)

    while True:
        state = node_queue.pop(0)  # pop out the state currently working on
        if len(path_queue) != 0:  # prepare for future pop of path
            path = path_queue.pop(0)

        if problem.isGoalState(state):  # if reach a goal state, end this while-loop
            return path

        visited.append(state)  # add the state that has already expanded
        successors = problem.getSuccessors(state)  # get the list of successors

        if len(successors) != 0:
            for node in successors:
                # This or statement is vital in some relatively complicated graph that repeats itself
                # for some nodes in different path. Avoid repetition in expanded states.
                if node[0] in visited or node[0] in node_queue:
                    continue
                else:
                    node_queue.append(node[0])
                    # it is vital to keep the concatenated form, or part of the path might be lose
                    # or the size of expanded nodes tend to be large
                    concatenated = path + [node[1]]
                    path_queue.append(concatenated)


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    start_state = problem.getStartState()  # the start state of the problem
    visited = []  # the states that already visited
    fringe = util.PriorityQueue()   # to deal with the minimum cost problem, apply priority queue

    fringe.push([start_state, [], 0], 0)    # concatenate all teh information together, it will be more convenient

    while True:
        state, path, cost = fringe.pop()  # pop out the state currently working on
        """
        Some testing code to show the change of variables:
        print (state)
        print (path)
        print (cost)
        """

        """
        This time my implementation involves an initial start with an empty list, 
        so it will never run into a case of NULL, no need to test about the 
        length of the path
        """
        if problem.isGoalState(state):  # if reach a goal state, end this while-loop
            return path

        """
        This "if-statement" is extremely important to avoid the state that is already reached before,
        since our priority is based on cost, if some state is already visited before, it means that 
        there has at least one path that is already less cost than this path, so skip it. 
        """
        if state in visited:
            continue

        successors = problem.getSuccessors(state)  # get the list of successors
        visited.append(state)  # add the state that has already expanded

        if len(successors) != 0:
            """
            Keep track of the information of all successors:
            # print(successors)
            """
            for node in successors:
                # This or statement is vital in some relatively complicated graph that repeats itself
                # for some nodes in different path. Avoid repetition in expanded states.
                if node[0] in visited:
                    continue
                else:
                    # it is vital to keep the concatenated form, or part of the path might be lose
                    # or the size of expanded nodes tend to be large
                    concatenated = path + [node[1]]
                    combined_cost = node[2] + cost
                    fringe.push([node[0], concatenated, combined_cost], combined_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()  # the start state of the problem
    visited = []  # the states that already visited
    fringe = util.PriorityQueue()  # to deal with the minimum cost problem, apply priority queue

    fringe.push([start_state, [], 0], 0)

    while True:
        state, path, cost = fringe.pop()  # pop out the state currently working on
        """
        Some testing code to show the change of variables:
        print (state)
        print (path)
        print (cost)
        """
        # print (state)
        # print (path)
        # print (cost)

        """
        This time my implementation involves an initial start with an empty list, 
        so it will never run into a case of NULL, no need to test about the 
        length of the path
        """
        if problem.isGoalState(state):  # if reach a goal state, end this while-loop
            return path

        """
        This "if-statement" is extremely important to avoid the state that is already reached before,
        since our priority is based on cost, if some state is already visited before, it means that 
        there has at least one path that is already less cost than this path, so skip it. 
        """
        if state in visited:
            continue

        successors = problem.getSuccessors(state)  # get the list of successors
        visited.append(state)  # add the state that has already expanded

        if len(successors) != 0:
            """
            Keep track of the information of all successors:
            # print(successors)
            """
            # print(successors)
            for node in successors:
                # This or statement is vital in some relatively complicated graph that repeats itself
                # for some nodes in different path. Avoid repetition in expanded states.
                if node[0] in visited:
                    continue
                else:
                    # it is vital to keep the concatenated form, or part of the path might be lose
                    # or the size of expanded nodes tend to be large

                    # The only difference here is the COST involves HEURISTIC
                    concatenated = path + [node[1]]
                    pure_cost = node[2] + cost
                    combined_cost = pure_cost + heuristic(node[0], problem)
                    fringe.push([node[0], concatenated, pure_cost], combined_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
