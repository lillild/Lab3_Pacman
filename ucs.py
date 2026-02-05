def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    
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
            print("UCS Solution Path:")
            print(actions)
            print("Total Cost:", cost_so_far)
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
    print("UCS: No solution found")
    return []
