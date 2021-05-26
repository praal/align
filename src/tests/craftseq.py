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
from environment import ReachFacts
from utils.report import SequenceReport
from rl.empathic_policy import Empathic
from environment.craft import OBJECTS

DEFAULT_Q = -1000.0

TOTAL_STEPS1 = 800000
TOTAL_STEPS2 = 2000000
TOTAL_STEPS3 = 600000
TOTAL_STEPS4 = 800000
TOTAL_STEPS5 = 2500000
EPISODE_LENGTH = 1000
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)

def evaluate_agents(env, rng1,policy1, reward1, env2, rng2, policy2, reward2, init, init2):
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
                print(s0, "----", a, "--->", s1)
                step_reward, finished = reward1(s0, a, s1)

                if not finished:
                    trial_reward += step_reward

                logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

                if finished:
                    print("final", s1)
                    break

            env2.reset(CraftState(initial_state2[0], initial_state2[1], env.state.key_x, env.state.key_y, set(), initial_state2[0], initial_state2[1]))
            print(env2.state, "!!!")
            print("(*****************")

            for step in range(EPISODE_LENGTH):

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


    rng1.setstate(rng_state)
    rng2.setstate(rng_state)

    return state_rewards, state_rewards2


def create_init(key_locations, init_locations):
    ans = []
    for i in key_locations:
        for j in init_locations:
            ans.append(CraftState(j[0], j[1], i[1], i[0], set(), j[0], j[1]))
            print("init: ", CraftState(j[0], j[1], i[1], i[0], set(), j[0], j[1]).uid)
    return ans

def visualize(results, labels, num_exp=5):
    results_list = []
    alpha = []
    x = []
    for key, value in results[0].items():
        alpha.append(key)
        x.append(key/100.0)
    print("alpha", alpha)
    for i in range(len(results)):
        results_list.append([])
        for a in alpha:
            results_list[i].append(results[i][a])
    for i in range(len(results_list)):
        print("***", results_list[i])
    for i in range(len(results_list)):
        means = np.mean(results_list[i], axis=1)
        for j in range(len(results_list[i])):
            for k in range(len(results_list[i][j])):
                if results_list[i][j][k] > 2.0 * means[j]:
                    pass

    folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
    os.makedirs(os.path.join("../datasets/", folder_name))
    ret_dataset = pd.DataFrame(results_list)
    ret_dataset.to_csv(os.path.join("../datasets/", folder_name, f"seq.csv"))

    for i in range(len(results_list)):
        means = np.mean(results_list[i], axis=1)

        std = np.std(results_list[i], axis=1)
        plt.plot(x, means, label=labels[i])
        ci = 1.96 * std / np.sqrt(num_exp)
        plt.fill_between(x, (means - ci), (means + ci), alpha=.1)


    plt.xlabel("Caring Factor (Alpha)")
    plt.ylabel("Average Reward")
    plt.legend(loc='best')

    plt.savefig(os.path.join("../datasets/", folder_name, f"seq-fig.png"))
    plt.show()

def run(filename, seed):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/map_seq.map")

    rng = Random(seed)
    env = Craft(map_fn, rng, 1, 1)
    print(env.get_all_item())
    init = create_init(env.get_all_item(), [[1,1]])

    tasks = [[OBJECTS["box"]]]
    not_task = [OBJECTS["key"]]
    not_task = []
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:
        emp_reward = {}
        selfish_reward = {}
        all_reward = {}
        for random_seed in range(10, 11):
            print("$$$$$$$$$$$", random_seed)
            print("ql: begin experiment")
            report = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)

            for j, goal in enumerate(tasks):
                print("ql: begin task {}".format(j + START_TASK))
                rng.seed(seed + j + random_seed)

                reward = ReachFacts(env, goal, not_task)
                policy1 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
                                       default_q=DEFAULT_Q, num_actions=4, rng=rng)
                agent = Agent(env, policy1, reward, rng)


                try:
                    start = time()
                    agent.train(steps=TOTAL_STEPS1,
                                steps_per_episode=EPISODE_LENGTH, report=report)


                    for alpha in range(200, 201):
                        #if alpha % 5 != 0:
                         #   continue
                        print("alpha:", alpha /100.0)
                        report2 = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)
                        rng2 = Random(seed+random_seed+ 1)
                        env2 = Craft(map_fn, rng2, 1, 1)
                        reward2 = ReachFacts(env2, goal, not_task)
                        policy2 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.4,
                                   default_q=DEFAULT_Q, num_actions=4, rng=rng2, others_q=policy1.get_Q(), penalty=-2*EPISODE_LENGTH, others_alpha=alpha / 100.0)
                        agent2 = Agent(env2, policy2, reward2, rng2)



                        start = time()
                        if alpha < 100:
                            agent2.train(steps=TOTAL_STEPS2,
                                steps_per_episode=EPISODE_LENGTH, report=report2)
                        elif alpha < 150:
                            agent2.train(steps=TOTAL_STEPS3,
                                         steps_per_episode=EPISODE_LENGTH, report=report2)
                        elif alpha < 200:
                            agent2.train(steps=TOTAL_STEPS4,
                                         steps_per_episode=EPISODE_LENGTH, report=report2)

                        else:
                            agent2.train(steps=TOTAL_STEPS5,
                                        steps_per_episode=EPISODE_LENGTH, report=report2)
                        end = time()
                        print(policy2.Q)
                        print("ql: Ran task {} for {} seconds.".format(j + START_TASK,end - start))
                        emp, sel = evaluate_agents(env2, rng2,policy2, reward2, env, rng, policy1, reward, [init[0]], [[1, 1]])

                        if alpha not in emp_reward:
                            emp_reward[alpha] = []
                            selfish_reward[alpha] = []
                            all_reward[alpha] = []
                        emp_reward[alpha].append(emp[0])
                        selfish_reward[alpha].append(sel[0])
                        all_reward[alpha].append((emp[0] + sel[0]) / 2)


                except KeyboardInterrupt:
                    end = time()
                    logging.warning("ql: interrupted task %s after %s seconds!",
                                    j + START_TASK, end - start)
        visualize([emp_reward, selfish_reward, all_reward], ["first agent", "second agent", "sum"])





def vis_test(labels = ["first agent", "second agent", "sum"]):
    x = []
    for i in range(0, 20):
        x.append(i/10.0)
    print("alpha", x)
    results_list = [[[-14.0, -12.0, -12.0, -21.0, -27.0, -12.0, -257.0, -14.0, -13.0, -12.0] , [-12.0, -12.0, -13.0, -12.0, -12.0, -12.0, -605.0, -12.0, -15.0, -13.0] , [-19.0, -14.0, -14.0, -12.0, -18.0, -17.0, -12.0, -12.0, -13.0, -13.0] , [-12.0, -13.0, -14.0, -12.0, -12.0, -12.0, -12.0, -13.0, -13.0, -20.0] , [-13.0, -13.0, -13.0, -12.0, -12.0, -15.0, -16.0, -12.0, -13.0, -13.0] , [-74.0, -13.0, -13.0, -12.0, -12.0, -12.0, -283.0, -13.0, -12.0, -13.0] , [-13.0, -139.0, -13.0, -13.0, -15.0, -12.0, -17.0, -12.0, -21.0, -13.0] , [-13.0, -13.0, -15.0, -12.0, -13.0, -13.0, -15.0, -13.0, -133.0, -12.0] , [-13.0, -15.0, -16.0, -12.0, -19.0, -12.0, -428.0, -13.0, -13.0, -19.0] , [-16.0, -13.0, -13.0, -16.0, -23.0, -31.0, -13.0, -13.0, -16.0, -13.0] , [-13.0, -15.0, -13.0, -16.0, -13.0, -13.0, -13.0, -12.0, -13.0, -15.0] , [-20.0, -16.0, -13.0, -16.0, -13.0, -12.0, -15.0, -12.0, -15.0, -16.0] , [-18.0, -15.0, -19.0, -16.0, -13.0, -15.0, -17.0, -13.0, -13.0, -13.0] , [-16.0, -13.0, -13.0, -18.0, -13.0, -13.0, -18.0, -15.0, -16.0, -16.0] , [-19.0, -13.0, -13.0, -16.0, -13.0, -20.0, -13.0, -13.0, -13.0, -20.0] , [-13.0, -15.0, -16.0, -16.0, -173.0, -13.0, -15.0, -13.0, -15.0, -16.0] , [-13.0, -478.0, -16.0, -16.0, -16.0, -13.0, -16.0, -16.0, -22.0, -16.0] , [-16.0, -16.0, -13.0, -16.0, -16.0, -16.0, -16.0, -13.0, -382.0, -16.0] , [-18.0, -417.0, -16.0, -169.0, -13.0, -16.0, -16.0, -18.0, -16.0, -18.0] , [-19.0, -13.0, -16.0, -26.0, -13.0, -159.0, -13.0, -16.0, -16.0, -16.0]],[[-18.0, -19.0, -20.0, -16.0, -16.0, -16.0, -16.0, -898.0, -15.0, -16.0] , [-18.0, -19.0, -14.0, -16.0, -18.0, -16.0, -14.0, -822.0, -14.0, -17.0] , [-14.0, -19.0, -22.0, -16.0, -16.0, -14.0, -16.0, -77.0, -14.0, -16.0] , [-18.0, -15.0, -14.0, -17.0, -16.0, -19.0, -16.0, -18.0, -14.0, -17.0] , [-14.0, -15.0, -14.0, -16.0, -16.0, -16.0, -16.0, -660.0, -14.0, -23.0] , [-14.0, -15.0, -14.0, -16.0, -16.0, -16.0, -14.0, -96.0, -16.0, -18.0] , [-14.0, -15.0, -14.0, -16.0, -14.0, -16.0, -26.0, -541.0, -16.0, -16.0] , [-14.0, -17.0, -14.0, -16.0, -14.0, -14.0, -14.0, -378.0, -14.0, -16.0] , [-14.0, -15.0, -12.0, -16.0, -15.0, -16.0, -12.0, -34.0, -14.0, -18.0] , [-14.0, -15.0, -14.0, -12.0, -14.0, -14.0, -14.0, -20.0, -14.0, -22.0] , [-14.0, -15.0, -14.0, -12.0, -14.0, -14.0, -14.0, -271.0, -16.0, -16.0] , [-12.0, -17.0, -14.0, -12.0, -14.0, -16.0, -14.0, -127.0, -14.0, -13.0] , [-12.0, -15.0, -14.0, -12.0, -16.0, -17.0, -24.0, -20.0, -14.0, -17.0] , [-12.0, -15.0, -14.0, -12.0, -14.0, -16.0, -13.0, -964.0, -14.0, -13.0] , [-12.0, -15.0, -14.0, -12.0, -14.0, -14.0, -16.0, -921.0, -14.0, -13.0] , [-14.0, -15.0, -12.0, -12.0, -12.0, -14.0, -14.0, -718.0, -14.0, -15.0] , [-14.0, -13.0, -12.0, -12.0, -12.0, -16.0, -12.0, -14.0, -12.0, -15.0] , [-12.0, -13.0, -14.0, -12.0, -12.0, -14.0, -12.0, -682.0, -16.0, -15.0] , [-12.0, -13.0, -12.0, -12.0, -14.0, -12.0, -12.0, -16.0, -12.0, -19.0] , [-14.0, -15.0, -12.0, -12.0, -14.0, -16.0, -14.0, -30.0, -12.0, -13.0]],[[-16.0, -15.5, -16.0, -18.5, -21.5, -14.0, -136.5, -456.0, -14.0, -14.0] , [-15.0, -15.5, -13.5, -14.0, -15.0, -14.0, -309.5, -417.0, -14.5, -15.0] , [-16.5, -16.5, -18.0, -14.0, -17.0, -15.5, -14.0, -44.5, -13.5, -14.5] , [-15.0, -14.0, -14.0, -14.5, -14.0, -15.5, -14.0, -15.5, -13.5, -18.5] , [-13.5, -14.0, -13.5, -14.0, -14.0, -15.5, -16.0, -336.0, -13.5, -18.0] , [-44.0, -14.0, -13.5, -14.0, -14.0, -14.0, -148.5, -54.5, -14.0, -15.5] , [-13.5, -77.0, -13.5, -14.5, -14.5, -14.0, -21.5, -276.5, -18.5, -14.5] , [-13.5, -15.0, -14.5, -14.0, -13.5, -13.5, -14.5, -195.5, -73.5, -14.0] , [-13.5, -15.0, -14.0, -14.0, -17.0, -14.0, -220.0, -23.5, -13.5, -18.5] , [-15.0, -14.0, -13.5, -14.0, -18.5, -22.5, -13.5, -16.5, -15.0, -17.5] , [-13.5, -15.0, -13.5, -14.0, -13.5, -13.5, -13.5, -141.5, -14.5, -15.5] , [-16.0, -16.5, -13.5, -14.0, -13.5, -14.0, -14.5, -69.5, -14.5, -14.5] , [-15.0, -15.0, -16.5, -14.0, -14.5, -16.0, -20.5, -16.5, -13.5, -15.0] , [-14.0, -14.0, -13.5, -15.0, -13.5, -14.5, -15.5, -489.5, -15.0, -14.5] , [-15.5, -14.0, -13.5, -14.0, -13.5, -17.0, -14.5, -467.0, -13.5, -16.5] , [-13.5, -15.0, -14.0, -14.0, -92.5, -13.5, -14.5, -365.5, -14.5, -15.5] , [-13.5, -245.5, -14.0, -14.0, -14.0, -14.5, -14.0, -15.0, -17.0, -15.5] , [-14.0, -14.5, -13.5, -14.0, -14.0, -15.0, -14.0, -347.5, -199.0, -15.5] , [-15.0, -215.0, -14.0, -90.5, -13.5, -14.0, -14.0, -17.0, -14.0, -18.5] , [-16.5, -14.0, -14.0, -19.0, -13.5, -87.5, -13.5, -23.0, -14.0, -14.5]]]

    for i in range(len(results_list)):
        print("***", results_list[i])
    result_ans = []
    for i in range(len(results_list)):
        means = np.mean(results_list[i], axis=1)
        result_ans.append([])
        for j in range(len(results_list[i])):
            result_ans[i].append([])
            for k in range(len(results_list[i][j])):
                if results_list[i][j][k] < 1.5 * means[j]:
                    continue
                result_ans[i][j].append(results_list[i][j][k])

    for i in range(len(result_ans)):
        means = []
        std = []
        for j in range(len(result_ans[i])):
            means.append(np.mean(result_ans[i][j]))
            std.append(np.std(result_ans[i][j]))

        means = np.array(means)
        std = np.array(std)
        plt.plot(x, means, label=labels[i])
        ci = 1.96 * std / np.sqrt(10)
        plt.fill_between(x, (means - ci), (means + ci), alpha=.1)


    folder_name = datetime.now().strftime("%d%m%Y_%H%M%S")
    os.makedirs(os.path.join("../datasets/", folder_name))
    ret_dataset = pd.DataFrame(result_ans)
    ret_dataset.to_csv(os.path.join("../datasets/", folder_name, f"seq.csv"))

    plt.xlabel("Caring Factor (Alpha)")
    plt.ylabel("Average Reward")
    plt.legend(loc='best')

    plt.savefig(os.path.join("../datasets/", folder_name, f"seq-fig.png"))
    plt.show()

run("../../../data/test.csv", 2019)
#vis_test()