# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import numpy as np

from voxel_mapping.voxel.voxel import SparseVoxelMap
from voxel_mapping.motion.base import Planner, PlanResult
from voxel_mapping.motion.space import ConfigurationSpace


def plan_to_frontier(
    start: np.ndarray,
    planner: Planner,
    space,
    visualize: bool = False,
    try_to_plan_iter: int = 10,
    debug: bool = False,
    verbose: bool = False,
    expand_frontier_size: int = 10,
) -> PlanResult:
    """Simple helper function for planning to the frontier during exploration.

    Args:
        start(np.ndarray): len=3 array containing world x, y, and theta
        planner(Planner): what we will use to generate motion plans to frontier points
    """
    # extract goal using fmm planner
    tries = 0
    failed = False
    res = None
    start_is_valid = space.is_valid(start, verbose=verbose)
    print("\n----------- Planning to frontier -----------")
    print("Starting at:", start)
    print("Start is valid:", start_is_valid)
    if not start_is_valid:
        return PlanResult(False, reason="invalid start state")
    
    for goal in space.sample_closest_frontier(
        start, verbose=verbose, debug=debug, expand_size=expand_frontier_size, min_dist=10.0,
    ):
    # for goal in space.sample_random_frontier():
        if goal is None:
            failed = True
            break
        goal = goal.cpu().numpy()
        print("       Start:", start)
        print("Sampled Goal:", goal)
        show_goal = np.zeros(3)
        show_goal[:2] = goal[:2]
        goal_is_valid = space.is_valid(goal)
        print("Start is valid:", start_is_valid)
        print(" Goal is valid:", goal_is_valid)
        if not goal_is_valid:
            print(" -> resample goal.")
            continue
        # plan to the sampled goal
        res = planner.plan(start, goal)
        print("Found plan:", res.success)

        if res.success:
            break
        else:
            if visualize:
                plt.show()
            tries += 1
            if tries >= try_to_plan_iter:
                failed = True
                break
            continue
    else:
        print(" ------ no valid goals found!")
        failed = True
    if failed:
        print(" ------ sampling and planning failed! Might be no where left to go.")
        return PlanResult(False, reason="planning to frontier failed")
    return res, goal

def plan_to_goal(
    start: np.ndarray,
    goal: np.ndarray,
    planner: Planner,
    space,
    verbose: bool = False,
) -> PlanResult:
    """Simple helper function for planning to the frontier during exploration.

    Args:
        start(np.ndarray): len=3 array containing world x, y, and theta
        planner(Planner): what we will use to generate motion plans to frontier points
    """
    res = None
    start_is_valid = space.is_valid(start, verbose=verbose)
    print("\n----------- Planning to frontier -----------")
    print("Starting at:", start)
    print("Start is valid:", start_is_valid)
    # if not start_is_valid:
    #     return PlanResult(False, reason="invalid start state")

    print("       Start:", start)
    print("       Goal:", goal)
    goal_is_valid = space.is_valid(goal)
    print("Start is valid:", start_is_valid)
    print("Goal is valid:", goal_is_valid)
    if not goal_is_valid:
        return PlanResult(False, reason="Goal is invalid")
    
    # plan to the sampled goal
    res = planner.plan(start, goal)
    print("Found plan:", res.success)

    return res