from bisect import bisect_left

import heapq
import random
import numpy as np

# Algorithm for truss growth, initial seeding
nodes = []
DOF = 3


def objective(x):
    return 1


weight = np.zeros_like(nodes)
for (i, node) in enumerate(nodes):
    weight[i] = weight[i - 1] + objective(node)

target = random.uniform(0, weight[-1])
closest = nodes[bisect_left(weight, target)]
k = closest.grow_node(objective)
top = k.get_within(2, 4)
rank = np.zeros_like(top)
for (i, l) in enumerate(top):
    rank[i] = (k - l).cross(k - closest).length / (k - closest).lengthsquared
heapq.heapify(zip(-rank, top))
for j in range(DOF - 1):
    k.connect(heapq.heappop()[1])

# Connect node close to objective to objective
