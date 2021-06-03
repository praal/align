import sys  # noqa
from datetime import datetime
from os import path as p  # noqa
import os
import time
import numpy as np


from rl.qvalue import EpsilonGreedy
from rl.rl import Agent
import matplotlib.pyplot as plt
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa
import csv
import logging
from os import path
from random import Random
from time import time
from typing import List, Tuple


from environment import ReachFacts
from environment.craft import Craft, CraftState, OBJECTS4, update_facts
from utils.report import SequenceReport
from rl.empathic_policy import Empathic


DEFAULT_Q = -1000.0

TOTAL_STEPS1 = 100000
TOTAL_STEPS2 = 300000

EPISODE_LENGTH = 1000
TEST_EPISODE_LENGTH = 50
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)
problem_mood = 4

def evaluate_agent(env, policy1, reward1, init):
    print("Evaluation:")
    state_rewards = []
    for initial_state1 in init:
        env.reset(initial_state1)
        reward1.reset()
        policy1.reset(evaluation=True)

        trial_reward: float = 0.0

        for step in range(TEST_EPISODE_LENGTH):
            s0 = env.state
            a = policy1.get_best_action(s0)
            env.apply_action(a)
            s1 = env.state
            print(s0, "----", a, "--->", s1)
            step_reward, finished = reward1(s0, a, s1)
            if not finished:
                trial_reward += step_reward
            logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)
            if finished:
                print("final", s1)
                break

        state_rewards.append(trial_reward)



def create_init(key_locations, init_locations):
    ans = []
    for i in key_locations:
        for j in init_locations:
            ans.append(CraftState(j[0], j[1], i[1], i[0], (), j[0], j[1], problem_mood))

    return ans

def create_options(key_locations, init_location):
    tmp = []

    options = []
    for loc in key_locations:
        ans = []
        tmp.append(loc)
        for t in tmp:
            ans.append(CraftState(init_location[0], init_location[1], t[1], t[0], (), init_location[0], init_location[1], problem_mood).uid)
        options.append(ans)
        if len(options) > 1:
            options.append(ans)
    return options

def train(filename, seed):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/options.map")

    rng1 = Random(seed + 1)
    env1 = Craft(map_fn, rng1, 7, 7, objects=OBJECTS4, problem_mood=problem_mood)
    print(env1.get_all_item())
    init1 = create_init([env1.get_all_item()[2]], [[7,7]])
    options = create_options(env1.get_all_item(), [7,7])
    tasks = [[OBJECTS4["target"]]]
    not_task = [OBJECTS4["key"]]
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:

        print("ql: begin experiment")
        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))

            try:
                start = time()
                report1 = SequenceReport(csvfile, LOG_STEP, init1, EPISODE_LENGTH, TRIALS)
                reward1 = ReachFacts(env1, goal, not_task, problem_mood)
                policy1 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.2,
                                    default_q=DEFAULT_Q, num_actions=4, rng=rng1, others_q=[], others_init=[[7,7], [7,7], [7,7], [7,7], [7,7]], others_dist=[0.2, 0.2, 0.2, 0.2, 0.2], penalty=-2*EPISODE_LENGTH, others_alpha=[50.0], objects=OBJECTS4, problem_mood = problem_mood, options = options)
                agent1 = Agent(env1, policy1, reward1, rng1)

                agent1.train(steps=TOTAL_STEPS2,
                              steps_per_episode=EPISODE_LENGTH, report=report1)
                evaluate_agent(env1, policy1, reward1, init1)

            except KeyboardInterrupt:
                end = time()
                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)



train("./test.csv", 2019)
