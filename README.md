# hierarchical-metric-proj
Utilization of hierarchical metrics for learning on data with a tree structure.


# TODO

## Notes for 18 October
- we will start with a toy problem in metric learning, e.g., two-class problems where data from each class form a line in 2d space.
- the task is to learn the right metric, i.e. push the lines sufficiently far from each other
- in the long run, we would like to learn that one of the dimensions is redundant
- as a first step, we will learn the metric using triplet loss which should be good enough [1]

TODO (add 'x' inside the brackets when done):
- [ ] write a basic description of the toy problem and corresponding code genertaing data
- [ ] adapt code from https://anonymous.4open.science/r/HTDExperiments/scripts/ContrastiveLearning.jl to run triplet loss on the toy data




References:
[1] Musgrave, K., Belongie, S. and Lim, S.N., 2020. A metric learning reality check. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXV 16 (pp. 681-699). Springer International Publishing.
