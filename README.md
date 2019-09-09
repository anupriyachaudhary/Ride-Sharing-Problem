# Ride Sharing Problem

## Problem Statement

#### Input:
- M requests with a pick-up and drop-off location
- N drivers, each with a starting location
- Fully connected weighted undirected graph of driving times
#### Constraints:
- Driver picks up at most two users, and then drops them off
- Enough drivers to pick up all users (m ≤ 2n)
- Every user must be assigned to a driver and brought to their destination
#### Goal:
Find a feasible assignment and driving paths that minimize the total driving time



## Complexity
Finding an optimal solution is NP-hard. A simplified version of this problem can be reduced to 3-dimensional perfect matching, a well known NP-hard problem


## Approximation Algorithm

### Cost Function:
Cost of matching rider i and rider j with driver k, 
cost (k, {i, j}) =  min {w(dk, si, sj, ti, tj), w(dk, si, sj, ti, tj), w(dk, si, sj, ti, tj), w(dk, si, sj, ti, tj)}

Cost of allocation M =  ∑ cost (k, RK)

### Greedy Algorithm:
##### Phase-1: Match 2n requests into n pairs
- Minimum weighted perfect matching for a Non Bipartite graph
- Edmond’s Blossom algorithm
##### Phase-2: Assign drivers to pairs
- Minimum weighted perfect matching in a Weighted Bipartite Graph

### Theoretical guarantee:
Cost(M) at most 2.5 times the optimal, O(n3)

## References
- http://www.ntu.edu.sg/home/xhbei/papers/ridesharing.pdf
- https://arxiv.org/pdf/1412.1130.pdf
- http://utc.mit.edu/uploads/MITR25-6-FP.pdf
- https://towardsdatascience.com/uber-driver-schedule-optimization-62879ea41658
- “Efficient Algorithms for Finding Maximum Matching in Graphs”, Zvi Galil, ACM Computing Surveys, 1986.


