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
from environment.craft import Craft, CraftState, OBJECTS2, update_facts
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
problem_mood = 2

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
    print("@@@@@@@@@@", key_locations)
    ans = []
    for i in key_locations:
        for j in init_locations:
            ans.append(CraftState(j[0], j[1], i[1], i[0], (), j[0], j[1], problem_mood))

    return ans


def train(filename, seed):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/doll.map")

    rng1 = Random(seed + 1)
    env1 = Craft(map_fn, rng1, 1, 5, objects=OBJECTS2, problem_mood=problem_mood)


    rng2 = Random(seed + 2)
    env2 = Craft(map_fn, rng2, 4, 1, objects=OBJECTS2, problem_mood=problem_mood)

    rng3 = Random(seed + 3)
    env3 = Craft(map_fn, rng3, 5, 2, objects=OBJECTS2, problem_mood=problem_mood)

    rng4 = Random(seed + 4)
    env4 = Craft(map_fn, rng4, 1, 3, objects=OBJECTS2, problem_mood=problem_mood)

    init1 = create_init([env1.get_all_item()[1]], [[1,5]])
    init2 = create_init(env2.get_all_item(), [[4,1]])
    init3 = create_init(env3.get_all_item(), [[5,2]])
    init4 = create_init(env4.get_all_item(), [[1,3]])


    tasks = [[OBJECTS2["played"]]]
    not_task = [OBJECTS2["doll"]]
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:

        print("ql: begin experiment")
        report1 = SequenceReport(csvfile, LOG_STEP, init1, EPISODE_LENGTH, TRIALS)
        report2 = SequenceReport(csvfile, LOG_STEP, init2, EPISODE_LENGTH, TRIALS)
        report3 = SequenceReport(csvfile, LOG_STEP, init3, EPISODE_LENGTH, TRIALS)
        report4 = SequenceReport(csvfile, LOG_STEP, init4, EPISODE_LENGTH, TRIALS)

        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))
            rng2.seed(seed + j)

            reward2 = ReachFacts(env2, goal, [], problem_mood)
            policy2 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.3,
                                    default_q=DEFAULT_Q, num_actions=4, rng=rng2)
            agent2 = Agent(env2, policy2, reward2, rng2)


            rng3.seed(seed + j)
            policy3 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.3,
                                default_q=DEFAULT_Q, num_actions=4, rng=rng3)
            agent3 = Agent(env3, policy3, reward2, rng3)

            rng4.seed(seed + j)
            policy4 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.3,
                                    default_q=DEFAULT_Q, num_actions=4, rng=rng4)
            agent4 = Agent(env4, policy4, reward2, rng4)

            try:
                start = time()
                agent2.train(steps=TOTAL_STEPS1,
                            steps_per_episode=EPISODE_LENGTH, report=report2)
                agent3.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report3)

                agent4.train(steps=TOTAL_STEPS1,
                             steps_per_episode=EPISODE_LENGTH, report=report4)

                report1 = SequenceReport(csvfile, LOG_STEP, init1, EPISODE_LENGTH, TRIALS)
                reward1 = ReachFacts(env1, goal, not_task, problem_mood)
                policy1 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.3,
                                    default_q=DEFAULT_Q, num_actions=5, rng=rng1, others_q=[policy2.get_Q(), policy2.get_Q(), policy3.get_Q(), policy3.get_Q(), policy4.get_Q()], others_init=[[4,1], [4, 1], [5, 2], [5, 2], [1, 3]], others_dist=[0.2, 0.2, 0.2, 0.2, 0.2], penalty=-2*EPISODE_LENGTH, others_alpha=[10.0, 10.0, 10.0, 10.0, 10.0], objects=OBJECTS2, problem_mood = problem_mood, caring_func="min")
                agent1 = Agent(env1, policy1, reward1, rng1)

                agent1.train(steps=TOTAL_STEPS2,
                              steps_per_episode=EPISODE_LENGTH, report=report1)
                evaluate_agent(env1, policy1, reward1, init1)

            except KeyboardInterrupt:
                end = time()
                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)



train("./test.csv", 2019)

