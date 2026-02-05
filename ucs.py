def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """
    Uniform Cost Search (UCS)

    Frontier:
    - Implemented using a priority queue (util.PriorityQueue)
    - Nodes are ordered by total path cost g(n)

    Graph Search:
    - Uses a dictionary (best_cost) to store the lowest cost found so far
      for each state to avoid revisiting more expensive paths
    """

    # Frontier stores tuples of:
    # (current_state, actions_to_reach_state, total_cost_g)
    frontier = util.PriorityQueue()

    # Start state of the problem
    start = problem.getStartState()

    # Push the start state into the frontier
    # g(start) = 0, so its priority is 0
    frontier.push((start, [], 0), 0)

    # best_cost[state] = lowest cost found so far to reach that state
    best_cost = {start: 0}

    # Continue searching while there are nodes in the frontier
    while not frontier.isEmpty():

        # Pop the node with the lowest g-value (total path cost)
        state, actions, cost_so_far = frontier.pop()

        # Graph search check:
        # If this path is more expensive than the best known path,
        # skip expanding this node
        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        # Goal test:
        # If the current state is a goal, return the solution path
        if problem.isGoalState(state):
            return actions

        # Expand the current node by generating its successors
        for successor, action, stepCost in problem.getSuccessors(state):

            # Compute g-value for the successor:
            # g(successor) = g(current) + stepCost
            new_cost = cost_so_far + stepCost

            # If this path to the successor is cheaper than any found before
            if new_cost < best_cost.get(successor, float("inf")):

                # Update the best known cost for this successor
                best_cost[successor] = new_cost

                # Add successor to the frontier
                # Priority is the total path cost g(successor)
                frontier.push(
                    (successor, actions + [action], new_cost),
                    new_cost
                )

    # If no solution is found, return an empty path
    return []
