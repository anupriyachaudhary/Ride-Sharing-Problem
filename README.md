# Ride Sharing Problem

## Problem Statement

### Input:
- M requests with a pick-up and drop-off location
- N drivers, each with a starting location
- Fully connected weighted undirected graph of driving times
### Constraints:
- Driver picks up at most two users, and then drops them off
- Enough drivers to pick up all users (m ≤ 2n)
- Every user must be assigned to a driver and brought to their destination
### Goal:
Find a feasible assignment and driving paths that minimize the total driving time



## Complexity
Finding an optimal solution is NP-hard. A simplified version of this problem can be reduced to 3-dimensional perfect matching, a well known NP-hard problem


## Approximation Algorithm
### Phase-1: Match 2n requests into n pairs
-Minimum weighted perfect matching for a Non Bipartite graph
-Edmond’s Blossom algorithm
### Phase-2: Assign drivers to pairs
- Minimum weighted perfect matching in a Weighted Bipartite Graph

### Theoretical guarantee
Cost(M) at most 2.5 times the optimal, O(n3)

