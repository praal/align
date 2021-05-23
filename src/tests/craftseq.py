import sys  # noqa
from os import path as p  # noqa
from datetime import datetime
import os
import pandas as pd
import time
import numpy as np
from rl.qvalue import EpsilonGreedy
from rl.rl import Agent
import matplotlib.pyplot as plt
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa

import logging
from os import path
from random import Random
from time import time
from typing import List, Tuple

from environment import Craft, CraftState, ReachFacts
from utils.report import SequenceReport
from rl.empathic_policy import Empathic
from environment.craft import OBJECTS

DEFAULT_Q = 0.0

TOTAL_STEPS1 = 2000000
TOTAL_STEPS2 = 2500000
EPISODE_LENGTH = 1000
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)

def evaluate_agents(env, rng1,policy1, reward1, rng2, policy2, reward2, init):
    init2 = [[1, 1]]
    rng_state = rng1.getstate()
    state_rewards = []
    state_rewards2 = []
    for initial_state2 in init2:
        for initial_state1 in init:

            env.reset(initial_state1)
            reward1.reset()
            policy1.reset(evaluation=True)

            reward2.reset()
            policy2.reset(evaluation=True)

            trial_reward: float = 0.0
            trial_reward2: float = 0.0

            for step in range(EPISODE_LENGTH):

                s0 = env.state
                a = policy1.get_best_action(s0)
                env.apply_action(a)
                s1 = env.state
                step_reward, finished = reward1(s0, a, s1)

                if not finished:
                    trial_reward += step_reward

                logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

                if finished:
                    break

            print(env.state)
            env.reset(CraftState(initial_state2[0], initial_state2[1], env.state.iron_x, env.state.iron_y, set()))

            for step in range(EPISODE_LENGTH):

                s0 = env.state
                a = policy2.get_best_action(s0)
                env.apply_action(a)
                s1 = env.state
                step_reward, finished = reward2(s0, a, s1)

                if not finished:
                    trial_reward2 += step_reward

                logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

                if finished:
                    break


            state_rewards.append(trial_reward)
            state_rewards2.append(trial_reward2)


    rng1.setstate(rng_state)
    rng2.setstate(rng_state)
    print(state_rewards)
    print("***")
    print(state_rewards2)
    return state_rewards, state_rewards2


def create_init(iron_locations, init_locations):
    ans = []
    for i in iron_locations:
        for j in init_locations:
            ans.append(CraftState(j[0], j[1], i[1], i[0], set()))
    return ans

def visualize(results, x, labels, num_exp=3):
    for i in range(len(results)):
        means = np.mean(results[i], axis=1)

        std = np.std(results[i], axis=1)
        plt.plot(x, means, label=labels[i])
        ci = 1.96 * std / np.sqrt(num_exp)
        plt.fill_between(x, (means - ci), (means + ci), alpha=.1)


    folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
    os.makedirs(os.path.join("../datasets/", folder_name))
    ret_dataset = pd.DataFrame(results)
    ret_dataset.to_csv(os.path.join("../datasets/", folder_name, f"seq.csv"))

    plt.xlabel("Caring Factor (Alpha)")
    plt.ylabel("Average Reward")
    plt.legend(loc='best')

    plt.savefig(os.path.join("../datasets/", folder_name, f"seq-fig.png"))
    plt.show()

def run(filename, seed):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/map_seq.map")

  #  init = [CraftState(1, 1, 2, 1, set()), CraftState(1, 1, 5, 5, set()), CraftState(1, 1, 4, 2, set())]
      #      CraftState(1, 1, 4, 3,set()), CraftState(1, 1, 3, 2,set()), CraftState(1, 1, 4, 4, set())]

    rng = Random(seed)

    env = Craft(map_fn, rng)

    init = create_init(env.get_all_item(), [[1,1]])

    tasks = [[OBJECTS["ring"]]]
    not_task = [OBJECTS["iron"]]
    #not_task = []
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:
        print("ql: begin experiment")
        report = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)

        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))
            rng.seed(seed + j)

            reward = ReachFacts(env, goal, not_task)
            policy1 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
                                   default_q=DEFAULT_Q, num_actions=4, rng=rng)
            agent = Agent(env, policy1, reward, rng)




        try:
                start = time()
                agent.train(steps=TOTAL_STEPS1,
                            steps_per_episode=EPISODE_LENGTH, report=report)

                end = time()
                print("ql: Ran task {} for {} seconds.".format(j + START_TASK,end - start))

                emp_reward = []
                self_reward = []
                all_reward = []
                alpha = []
                for a in range(5, 20):
                    if a %  2 == 0:
                        continue
                    print("alpha:", a /10.0)
                    report2 = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)
                    rng2 = Random(seed+a)
                    env2 = Craft(map_fn, rng2)
                    reward2 = ReachFacts(env, goal, not_task)
                    policy2 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.1,
                               default_q=DEFAULT_Q, num_actions=4, rng=rng, others_q=policy1.get_Q(), penalty=-1*EPISODE_LENGTH, others_alpha=a / 10.0)
                    agent2 = Agent(env2, policy2, reward2, rng2)


                    start = time()
                    agent2.train(steps=TOTAL_STEPS2,
                            steps_per_episode=EPISODE_LENGTH, report=report2)

                    end = time()
                    print("ql: Ran task {} for {} seconds.".format(j + START_TASK,end - start))
                    emp, sel = evaluate_agents(env, rng2,policy2, reward2, rng, policy1, reward, init)
                    emp_reward.append(emp)
                    self_reward.append(sel)
                    all_reward.append(emp + sel)
                    alpha.append(a / 10.0)
                visualize([emp_reward, self_reward, all_reward], alpha, labels=["first agent", "second agent", "sum"])


        except KeyboardInterrupt:
                end = time()
                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)







run("../../../data/test.csv", 2019)