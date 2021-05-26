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
import csv
import logging
from os import path
from random import Random
from time import time
from typing import List, Tuple

from environment import Craft, CraftState
from environment import ReachFacts
from utils.report import SequenceReport
from rl.empathic_policy import Empathic
from environment.craft import OBJECTS, update_facts

DEFAULT_Q = -1000.0

TOTAL_STEPS1 = 1000000
TOTAL_STEPS2 = 2000000

EPISODE_LENGTH = 1000
TEST_EPISODE_LENGTH = 50
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)

def evaluate_agents(env, policy1, reward1, env2, policy2, reward2, init, init2):
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

            facts = set()
            if env.state.facts[4] == 1:
                facts.add(4)
            env2.reset(CraftState(initial_state2[0], initial_state2[1], env.state.key_x, env.state.key_y, facts, initial_state2[0], initial_state2[1]))
            print(env2.state, "!!!")
            print("(*****************")

            for step in range(TEST_EPISODE_LENGTH):
                s0 = env2.state
                a = policy2.get_best_action(s0)
                env2.apply_action(a)
                s1 = env2.state
                step_reward, finished = reward2(s0, a, s1)
                if not finished:
                    trial_reward2 += step_reward

                logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

                if finished:
                    break

            state_rewards.append(trial_reward)
            state_rewards2.append(trial_reward2)
    return state_rewards, state_rewards2


def create_init(key_locations, init_locations):
    ans = []
    for i in key_locations:
        for j in init_locations:
            ans.append(CraftState(j[0], j[1], i[1], i[0], (), j[0], j[1]))
            print("init: ", CraftState(j[0], j[1], i[1], i[0],(), j[0], j[1]).uid)
            facts, _ = update_facts((), {}, 0, True)
            ans.append(CraftState(j[0], j[1], i[1], i[0], facts, j[0], j[1]))
            print("init: ", CraftState(j[0], j[1], i[1], i[0],facts, j[0], j[1]).uid)
    return ans


def visualize(foldername, labels=["First Agent", "Second Agent", "Average"], num_exp=5):
    results = []
    alpha = []
    x = []
    for i in range(len(labels)):
        results.append([])
    for a in range(0, 100):
        if a % 10 != 0:
            continue
        alpha.append(a)
        x.append(a/100.0)
        with open(os.path.join("../datasets/", foldername, f"seq_{a}.csv")) as alpha_file:
            csv_reader = csv.reader(alpha_file, delimiter=',')
            i = 0
            for row in csv_reader:
                tmp = []
                for r in row:
                    tmp.append(float(r))
                results[i].append(tmp)
                i += 1


    for i in range(len(results)):
        means = np.mean(results[i], axis=1)
        std = np.std(results[i], axis=1)
        plt.plot(x, means, label=labels[i])
        print(x , means)
        ci = 1.96 * std / np.sqrt(num_exp)
        plt.fill_between(x, (means - ci), (means + ci), alpha=.1)


    plt.xlabel("Caring Factor (Alpha)")
    plt.ylabel("Average Reward")
    plt.legend(loc='best')

    plt.savefig(os.path.join("../datasets/", foldername, f"seq-fig.png"))
    plt.show()

def train(filename, seed, foldername):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/map_seq.map")

    rng1 = Random(seed)
    env1 = Craft(map_fn, rng1, 1, 1, tool_in_fact=True)
    print(env1.get_all_item())
    init = create_init(env1.get_all_item(), [[1,1]])

    tasks = [[OBJECTS["box"]]]
    not_task = [OBJECTS["key"]]
    not_task = []
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:

        print("ql: begin experiment")
        report = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)

        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))
            rng1.seed(seed + j)

            reward1 = ReachFacts(env1, goal, not_task)
            policy1 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.3,
                                   default_q=DEFAULT_Q, num_actions=4, rng=rng1)
            agent = Agent(env1, policy1, reward1, rng1)



            try:
                start = time()
                agent.train(steps=TOTAL_STEPS1,
                            steps_per_episode=EPISODE_LENGTH, report=report)

                #print(policy1.get_Q())
                for alpha in range(0, 100):
                    if alpha % 10 != 0:
                        continue
                    print("alpha:", alpha /100.0)
                    report2 = SequenceReport(csvfile, LOG_STEP, [init[0], init[2]], EPISODE_LENGTH, TRIALS)
                    rng2 = Random(seed+ 1)
                    env2 = Craft(map_fn, rng2, 1, 1)
                    reward2 = ReachFacts(env2, goal, not_task)
                    policy2 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.2,
                               default_q=DEFAULT_Q, num_actions=5, rng=rng2, others_q=policy1.get_Q(), penalty=-2*EPISODE_LENGTH, others_alpha=alpha / 100.0)
                    agent2 = Agent(env2, policy2, reward2, rng2)

                    agent2.train(steps=TOTAL_STEPS2,
                            steps_per_episode=EPISODE_LENGTH, report=report2)

                    test(env2, policy2, reward2, env1, policy1, reward1, init, alpha, foldername)

            except KeyboardInterrupt:
                end = time()
                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)
 #   visualize([emp_reward, selfish_reward, all_reward], ["first agent", "second agent", "sum"])


def test(env1, policy1, reward1, env2, policy2, reward2, init, alpha, foldername):

    emp_reward = []
    selfish_reward = []
    all_reward = []
    for seed in range(1):
        emp, sel = evaluate_agents(env1,policy1, reward1, env2, policy2, reward2, [init[0]], [[1, 1]])
        emp_reward.append(emp[0])
        selfish_reward.append(sel[0])
        all_reward.append((emp[0] + sel[0]) / 2)

    with open(os.path.join("../datasets/", foldername, f"seq_{alpha}.csv"), mode='w+') as alpha_file:
        writer = csv.writer(alpha_file, delimiter=',')
        writer.writerow(emp_reward)
        writer.writerow(selfish_reward)
        writer.writerow(all_reward)

folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
os.makedirs(os.path.join("../datasets/", folder_name))
train("../../../data/test.csv", 2019, folder_name)
visualize(folder_name)
#create_datasets("25052021_223024", 200)