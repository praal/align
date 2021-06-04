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
from environment.craft import Craft, CraftState, OBJECTS3, update_facts
from utils.report import SequenceReport
from rl.empathic_policy import Empathic


DEFAULT_Q = 0.0

TOTAL_STEPS1 = 100000
TOTAL_STEPS2 = 200000

EPISODE_LENGTH = 1000
TEST_EPISODE_LENGTH = 50
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)
problem_mood = 3

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



def create_init(key_locations, init_locations, fence=False):
    print("@@@@@@@@@@", key_locations)
    ans = []
    for i in key_locations:
        for j in init_locations:
            ans.append(CraftState(j[0], j[1], i[1], i[0], (), j[0], j[1], problem_mood))
            if fence:
                facts = set()
                facts.add(1)
                ans.append(CraftState(j[0], j[1], i[1], i[0], facts, j[0], j[1], problem_mood))

    return ans


def train(filename, seed, alpha2):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/garden.map")

    rng1 = Random(seed + 1)
    env1 = Craft(map_fn, rng1, 1, 5, objects=OBJECTS3, problem_mood=problem_mood)


    rng2 = Random(seed + 2)
    env2 = Craft(map_fn, rng2, 1, 5, objects=OBJECTS3, problem_mood=problem_mood, fence=True)


    init1 = create_init(env1.get_all_item(), [[1,5]])
    init2 = create_init(env1.get_all_item(), [[1,5]], True)




    tasks = [[OBJECTS3["target"]]]
    not_task = []
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:

        print("ql: begin experiment")
        report1 = SequenceReport(csvfile, LOG_STEP, init1, EPISODE_LENGTH, TRIALS)
        report2 = SequenceReport(csvfile, LOG_STEP, init2, EPISODE_LENGTH, TRIALS)


        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))
            rng2.seed(seed + j)

            reward2 = ReachFacts(env2, goal, [], problem_mood)
            policy2 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.2,
                                    default_q=DEFAULT_Q, num_actions=4, rng=rng2)
            agent2 = Agent(env2, policy2, reward2, rng2)


            try:
                start = time()
                agent2.train(steps=TOTAL_STEPS1, steps_per_episode=EPISODE_LENGTH, report=report2)

                reward1 = ReachFacts(env1, goal, not_task, problem_mood)
                policy1 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.2,
                                   default_q=DEFAULT_Q, num_actions=5, rng=rng1, others_q=[policy2.get_Q(), policy2.get_Q()], others_init=[[1,5], [1,5]], others_dist=[0.5, 0.5], penalty=-2*EPISODE_LENGTH, others_alpha=[alpha2, 1.0], objects=OBJECTS3, problem_mood = problem_mood, caring_func="sum", restricted=[[2, 1], []])
                agent1 = Agent(env1, policy1, reward1, rng1)

                agent1.train(steps=TOTAL_STEPS2,
                             steps_per_episode=EPISODE_LENGTH, report=report1)
                evaluate_agent(env1, policy1, reward1, init1)

            except KeyboardInterrupt:
                end = time()
                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)



start = time()
alpha2 = float(sys.argv[1])
train("./test.csv", 2019, alpha2)
end = time()
print("Total Time:", end - start)



