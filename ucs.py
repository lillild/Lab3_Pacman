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
