# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
GOAL_FOUND = False

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def recursiveDfs(current, visited, moves, problem):
    '''
    Recursive function for DepthFirstSearch algorithm

    :param current: (tuple) Current node
    :param visited: (list) Already visited nodes
    :param moves: (list) Moves conducted so far
    :param problem: (SearchProblem)

    :returns: (list) Moves needed to get to goal using DFS algorithm
    '''
    visited.append(current)
    if problem.isGoalState(current):
        global GOAL_FOUND
        GOAL_FOUND = True
        return

    successors = problem.getSuccessors(current)
    if successors:
        if (successors[0][1] == 'North' or successors[0][1] == 'East' or
            successors[0][1] == 'South' or successors[0][1] == 'West'):
            successors.sort(key = sortRule)
        else:
            successors.sort(key = lambda successors: successors[1], reverse = True)
    for move in successors:
        if move[0] not in visited:
            moves.push(move[1])
            recursiveDfs(move[0], visited, moves, problem)
        if GOAL_FOUND:
            return

    if GOAL_FOUND:
        return
    if not moves.isEmpty():
        moves.pop()
    return

def sortRule(item):
    if item[1] == 'West':
        return 0
    if item[1] == 'East':
        return 1
    if item[1] == 'South':
        return 2
    if item[1] == 'North':
        return 3


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    global GOAL_FOUND
    GOAL_FOUND = False
    moves = util.Stack()
    visited = []
    recursiveDfs(problem.getStartState(), visited, moves, problem)
    to_goal = []

    while not moves.isEmpty():
        item = moves.pop()
        to_goal.append(item)
    to_goal.reverse()

    return to_goal


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    visited = []
    queue = util.PriorityQueueWithFunction(bfsFunx)
    queue.push((problem.getStartState(),'',[''],0,0,0))

    while queue.isEmpty() == False:
        current = queue.pop()
        if current[0] in visited:
            continue
        visited.append(current[0])
        if problem.isGoalState(current[0]):
            break
        successors = [problem.getSuccessors(current[0])]

        for i, state in enumerate(successors[0]):
            go_to = [[]]
            go_to[0] = go_to[0] + current[2]
            if state in visited:
                continue
            else:
                go_to[0].append(state[1])
                item = [state[0], state[1], go_to[0], state[2]+current[3], state[2]+current[3]]
                queue.push(item)

    return current[2][1:]

def bfsFunx(item):
    '''
    For breadthFirstSearch algorithm
    :param item: (list)
    :return: item[3]
    '''
    return item[3]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visited = []
    cost = [1]
    queue = util.PriorityQueueWithFunction(ucsFunx)
    queue.push([problem.getStartState(), '', [''], 0, 0, 0])

    while queue.isEmpty() == False:
        current = queue.pop()
        if current[0] in visited:
            continue
        visited.append(current[0])
        if problem.isGoalState(current[0]):
            break
        successors = [problem.getSuccessors(current[0])]

        for i, state in enumerate(successors[0]):
            go_to = [[]]
            go_to[0] = go_to[0] + current[2]
            if state in visited:
                continue
            else:
                go_to[0].append(state[1])
                item = [state[0], state[1], go_to[0], state[2]+current[3], state[2]+current[3], cost[0]]
                queue.push(item)
                cost[0] -= 1
    return current[2][1:]

def ucsFunx(item):
    '''
    For uniformCostSearch algorithm
    :param item: (list)
    :return: item[4]
    '''
    return item[4]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visited = []
    cost = [1]
    queue = util.PriorityQueueWithFunction(ucsFunx)
    queue.push((problem.getStartState(),'',[''],0,0,0))

    while queue.isEmpty() == False:
        current = queue.pop()
        if current[0] in visited:
            continue
        visited.append(current[0])
        if problem.isGoalState(current[0]):
            break
        successors = [problem.getSuccessors(current[0])]

        for i,state in enumerate(successors[0]):
            state_heuristic = heuristic(state[0], problem)
            go_to = [[]]
            go_to[0] = go_to[0] + current[2]
            if state in visited:
                continue
            else:
                go_to[0].append(state[1])
                item = [state[0], state[1], go_to[0], state[2]+current[3],
                state[2] + current[3] + state_heuristic, cost[0]]
                queue.push(item)
                cost[0] -= 1
    return current[2][1:]



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
