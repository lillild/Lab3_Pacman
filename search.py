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

from math import dist
from tracemalloc import start
import util
from game import Directions
from typing import List

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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #node is tuple (state, actions)
    stack = util.Stack() #DFS uses a stack (LIFO)
    start = problem.getStartState()

    # push starting state to stack(state, empty list of actions)
    # create set to keep track of explored states
    stack.push((start, []))
    visited = set()

    while not stack.isEmpty():
        state, actions = stack.pop() #pop starting node/top node

        if problem.isGoalState(state): #finish search
            print(actions)
            return actions #return list of actions to reach goal for pac to go

        if state in visited: #push adj node (if visited, skip; if not add to visited)
            continue
        else:
            visited.add(state)

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                new_actions = actions + [action]  # create new list of actions
                stack.push((successor, new_actions))  # push (state, actions) 
    return []  # if no solution
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue() #BFS uses a queue (FIFO)
    start = problem.getStartState()

    # Store (state, actions_so_far)
    queue.push((start, []))
    visited = set()

    while not queue.isEmpty():
        state, actions = queue.pop()

        if problem.isGoalState(state):
            print(actions)
            return actions

        if state in visited:
            continue
        else:
            visited.add(state)

        for successor, action, stepCost in problem.getSuccessors(state):
            if successor not in visited:
                new_actions = actions + [action]  # create new list of actions
                queue.push((successor, new_actions))
    return []  # if no solution
    util.raiseNotDefined()

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.
    """
    start=problem.getStartState() #find initial pacman position
    if problem.isGoalState(start):
        return []

    def depth_limited_dfs(limit):
        #local function which returns list of actions if goal is found in the sub-branch (else returns None)
        stack = util.Stack() #stack = (state, path, depth)
        stack.push((start, [], 0))
        best_depth = {start: 0} #tracks smallest depth for 1 run

        while not stack.isEmpty():
            state, path, depth = stack.pop() #LIFO
            if problem.isGoalState(state):
                print(path)
                return path
            if depth == limit:
                continue #don't look into children if limit reached (pop next node from stack instead)
            
            for succ, action, stepCost in problem.getSuccessors(state):
                next_depth = depth + 1
                if succ not in best_depth or next_depth < best_depth[succ]: #if succ is new or there's an easier way to reach succ, push it
                    best_depth[succ] = next_depth
                    stack.push((succ, path + [action], next_depth))
        return None #if no solution found at this limit
    
    depth=1
    while True:
        result = depth_limited_dfs(depth)
        if result is not None:
            return result
        depth += 1
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()

    start = problem.getStartState()

   
    frontier.push((start, [], 0), 0)

    best_cost = {start: 0}

    while not frontier.isEmpty():

        state, actions, cost_so_far = frontier.pop()

        
        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        # Goal test
        if problem.isGoalState(state):
            print(actions)
            return actions

        for successor, action, stepCost in problem.getSuccessors(state):

            # Compute g-value for the successor
            new_cost = cost_so_far + stepCost

            if new_cost < best_cost.get(successor, float("inf")):

                # Update the best known cost
                best_cost[successor] = new_cost

                # Add successor to the frontier
                frontier.push(
                    (successor, actions + [action], new_cost),
                    new_cost
                )
    # If no solution is found, return an empty path
    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """
    A* graph search:
      - frontier is a PriorityQueue ordered by f(n) = g(n) + h(n)
      - g(n) = cost of actions so far
      - h(n) = heuristic(state, problem)
    """
def aStarSearch(problem, heuristic=nullHeuristic):
    pq = util.PriorityQueue() #priority queue for the min on top
    start = problem.getStartState()
    start_actions = []
    start_g = 0
    start_h = heuristic(start, problem)
    start_priority = start_g + start_h

    dist = {}              # best g(n) found so far
    dist[start] = 0

    # PQ stores (item, priority)
    pq.push((start, start_actions, start_g), start_priority)

    while not pq.isEmpty():
        state, actions, g = pq.pop()

        best_cost = dist.get(state, float("inf"))
        if g > best_cost: # If this entry is not the best one, skip expanding
            continue

        if problem.isGoalState(state): #goal check
            print(actions)
            return actions

        for successor, action, stepCost in problem.getSuccessors(state): #wehre to go next [('B', 'go_to_B', 2)]
            new_g = g + stepCost #new g is cost to reach current node + step cost to successor 0+2=2
            if new_g < dist.get(successor, float("inf")): #if never been to then inf
                dist[successor] = new_g #update b as 2, overlaping inf
                priority_num = new_g + heuristic(successor, problem)
                pq.push((successor, actions + [action], new_g), priority_num) #go from B now 

    return []
    util.raiseNotDefined()




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch  
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
