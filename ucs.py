def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first (Uniform Cost Search)."""

    frontier = util.PriorityQueue()
    start = problem.getStartState()

    # Each item in the PriortyQueue: (state, actions_so_far, cost_so_far)
    frontier.push((start, [], 0), 0)

    # best_cost[state] = cheapest cost found so far to reach state
    best_cost = {start: 0}

    while not frontier.isEmpty():
        state, actions, cost_so_far = frontier.pop()

        # If this is an outdated (more expensive) entry, skip it
        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        if problem.isGoalState(state):
            return actions

        for successor, action, stepCost in problem.getSuccessors(state):
            new_cost = cost_so_far + stepCost

            # Only consider this successor if it's a cheaper path than before
            if new_cost < best_cost.get(successor, float("inf")):
                best_cost[successor] = new_cost
                frontier.push((successor, actions + [action], new_cost), new_cost)

    # If no solution exists
    return []