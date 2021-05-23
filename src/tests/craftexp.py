import sys  # noqa
from os import path as p  # noqa

from rl.qvalue import EpsilonGreedy
from rl.rl import Agent

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

TOTAL_STEPS1 = 700000
TOTAL_STEPS2 = 1500000
EPISODE_LENGTH = 1000
LOG_STEP = 10000
TRIALS = 5
START_TASK = 0
END_TASK = 3
logging.basicConfig(level=logging.INFO)

def evaluate_agents( env1, rng1,policy1, reward1, env2, rng2, policy2, reward2):
    init = [CraftState(3, 3, set()), CraftState(7, 3, set()),
            CraftState(5, 5, set()), CraftState(3, 7, set()),
            CraftState(7, 7, set())]
    rng_state = rng1.getstate()
    state_rewards = []
    state_rewards2 = []
    for initial_state1 in init:
        for initial_state2 in init:
            if initial_state1 == initial_state2:
                continue
            trial_rewards: List[float] = []
            trial_rewards2: List[float] = []
            env1.reset(initial_state1)
            reward1.reset()
            policy1.reset(evaluation=True)

            env2.reset(initial_state2)
            reward2.reset()
            policy2.reset(evaluation=True)

            trial_reward: float = 0.0
            trial_reward2: float = 0.0
          #  print("start -------------------------------------------")
            for step in range(EPISODE_LENGTH):

                s0 = env1.state
                s0_2 = env2.state
                is_iron = True

                if env1.state.facts[1] or env2.state.facts[1]:
                    is_iron = False
               # print(s0)
               # print("**", is_iron)
               # print(s0_2)

                a = policy1.get_best_action(s0)
                a_2 = policy2.get_best_action(s0_2)


                env1.apply_action(a, is_iron)
                s1 = env1.state

                env2.apply_action(a_2, is_iron)
                s1_2 = env2.state

                step_reward, finished = reward1(s0, a, s1)
                step_reward2, finished2 = reward2(s0_2, a_2, s1_2)

                if not finished:
                    trial_reward += step_reward
                if not finished2:
                    trial_reward2 += step_reward2

                logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

                if finished and finished2:
                    break


            state_rewards.append(trial_reward)
            state_rewards2.append(trial_reward2)


    rng1.setstate(rng_state)
    rng2.setstate(rng_state)
    print(state_rewards)
    print("***")
    print(state_rewards2)
    return state_rewards, state_rewards2

def run(filename, seed):

    here = path.dirname(__file__)
    map_fn = path.join(here, "craft/map.map")

    init = [CraftState(3, 3, set()), CraftState(7, 3, set()),
            CraftState(5, 5, set()), CraftState(3, 7, set()),
            CraftState(7, 7, set())]

    rng = Random(seed)
    rng2 = Random(seed+1)
    env = Craft(map_fn, rng)
    env2 = Craft(map_fn, rng2)

    rng3 = Random(seed+2)
    env3 = Craft(map_fn, rng3)


    rng4 = Random(seed+2)
    env4 = Craft(map_fn, rng4)

    tasks = [[OBJECTS["ring"]]]
    not_task = []
    tasks = tasks[START_TASK:END_TASK+1]

    with open(filename, "w") as csvfile:
        print("ql: begin experiment")
        report = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)
        report2 = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)
        report3 = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)

        for j, goal in enumerate(tasks):
            print("ql: begin task {}".format(j + START_TASK))
            rng.seed(seed + j)
            rng2.seed(seed + 1 + j)
            rng3.seed(seed + 2 + j)

            reward = ReachFacts(env, goal, [])
            policy1 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
                                   default_q=DEFAULT_Q, num_actions=4, rng=rng)
            agent = Agent(env, policy1, reward, rng)
            reward3 = ReachFacts(env3, goal, [])
            policy3 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
                                    default_q=DEFAULT_Q, num_actions=4, rng=rng3)
            agent3 = Agent(env3, policy3, reward3, rng3)


            try:
                start = time()
                agent.train(steps=TOTAL_STEPS1,
                            steps_per_episode=EPISODE_LENGTH, report=report)


                agent3.train(steps=TOTAL_STEPS1,
                            steps_per_episode=EPISODE_LENGTH, report=report3)

                end = time()
                print("ql: Ran task {} for {} seconds.".format(j + START_TASK,
                                                               end - start))
            except KeyboardInterrupt:
                end = time()
                logging.warning("ql: interrupted task %s after %s seconds!",
                                j + START_TASK, end - start)

            reward2 = ReachFacts(env2, goal, not_task)
            policy2 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.1,default_q=DEFAULT_Q, num_actions=4, rng=rng2, others_q=policy1.get_Q(), penalty=-1*EPISODE_LENGTH, others_alpha=1)
           # policy2 = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
           #                         default_q=DEFAULT_Q, num_actions=4, rng=rng)

            reward4 = ReachFacts(env4, goal, not_task)
            policy4 = Empathic(alpha=1.0, gamma=1.0, epsilon=0.1,default_q=DEFAULT_Q, num_actions=4, rng=rng4, others_q=policy1.get_Q(), penalty=-1*EPISODE_LENGTH, others_alpha=1)


            agent2 = Agent(env2, policy2, reward2, rng2)
            agent2.train(steps=TOTAL_STEPS2,
                         steps_per_episode=EPISODE_LENGTH, report=report2)


            agent4 = Agent(env4, policy4, reward4, rng4)
            agent4.train(steps=TOTAL_STEPS2,
                         steps_per_episode=EPISODE_LENGTH, report=report2)
           # p2 = policy2.get_policy()
           # p1 = policy1.get_policy()

           # for state in p1:
           #     a1 = p1[state]
           #     a2 = p2[state]
            #    if a1 != a2:
             #       print(state, "********", a1, a2)


            report.increment(TOTAL_STEPS1)
            report2.increment(TOTAL_STEPS2)
            print(agent.evaluate(init, EPISODE_LENGTH, TRIALS))
            print("))))))))))))))))))))))))))))")
            evaluate_agents(env, rng,policy1, reward, env2, rng2, policy2, reward2)
            print("-------------------")

            evaluate_agents(env, rng,policy1, reward, env3, rng3, policy3, reward3)
            print("((((!!")
            evaluate_agents(env2, rng2,policy2, reward2, env4, rng4, policy4, reward4)

run("../../../data/test.csv", 2019)