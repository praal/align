import sys  # noqa
from os import path as p  # noqa
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa

import logging
from random import Random
from time import time
from typing import List, Tuple

from environment import Office, OfficeState, ReachFacts
from rl import Agent, EpsilonGreedy, HLAction, HRL, MonPOP, MonSeq, Option, \
        POPlan, SeqPlan, Shaped
from utils import SequenceReport

DEFAULT_Q = 0.0
TOTAL_STEPS = 5000000
EPISODE_LENGTH = 1000
LOG_STEP = 10000
TRIALS = 10
START_TASK = 10
END_TASK = 10

logging.basicConfig(level=logging.INFO)

states = []
for s in [(0, 0), (11, 8), (2, 6), (9, 2), (0, 4), (11, 4), (5, 8), (6, 0),
          (3, 4), (8, 4)]:
    states.append(OfficeState(s[0], s[1], []))

rng = Random(2019)
env = Office(rng)

tasks = [([0], 200000), ([1], 200000), ([2], 200000), ([3], 200000),
         ([4], 200000), ([5], 200000), ([6], 200000), ([7], 500000),
         ([8], 500000), ([7, 8], 1000000), ([0, 1, 2, 3], 1000000),
         ([0, 1, 2, 3, 7, 8], 3000000), ([0, 1, 2, 3, 4, 5, 6, 7, 8], 5000000)]
tasks = tasks[START_TASK:END_TASK+1]

with open("../../data/office-ql.csv", "w") as csvfile:
    print("ql: begin experiment")
    report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
    for i, (goal, steps) in enumerate(tasks):
        print("ql: begin task {}".format(i + START_TASK))
        print(goal)
        reward = ReachFacts(env, goal)
        policy = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
                               default_q=DEFAULT_Q, num_actions=4, rng=rng)
        agent = Agent(env, policy, reward, rng)
        try:
            start = time()
            agent.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
                        report=report)
            end = time()
            print("ql: Ran task {} for {} seconds.".format(i + START_TASK,
                                                           end - start))
        except KeyboardInterrupt:
            logging.warning("interrupted! (ql, task %s)", i + START_TASK)

        report.increment(TOTAL_STEPS)

del policy
del agent
#
# # go to A
# policyA = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionA = Option(env, policyA, lambda s: 'a' in env.observe(s))  # type: ignore
#
# # go to B
# policyB = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionB = Option(env, policyB, lambda s: 'b' in env.observe(s))  # type: ignore
#
# # go to C
# policyC = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionC = Option(env, policyC, lambda s: 'c' in env.observe(s))  # type: ignore
#
# # go to D
# policyD = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionD = Option(env, policyD, lambda s: 'd' in env.observe(s))  # type: ignore
#
# # get mail
# policyE = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionE = Option(env, policyE, lambda s: 'e' in env.observe(s))  # type: ignore
#
# # get coffee
# policyF = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionF = Option(env, policyF, lambda s: 'f' in env.observe(s))  # type: ignore
#
# # go to office
# policyG = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1, default_q=DEFAULT_Q,
#                         num_actions=4, rng=rng)
# optionG = Option(env, policyG, lambda s: 'g' in env.observe(s))  # type: ignore
#
# options = [optionA, optionB, optionC, optionD, optionE, optionF, optionG]
#
# with open("/tmp/office-hrl.csv", "w") as csvfile:
#     print("hrl: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("hrl: begin task {}".format(i + START_TASK))
#         reward = ReachFacts(env, goal)
#         policyHRL = HRL(EpsilonGreedy(alpha=0.1, gamma=1.0, epsilon=0.1,
#                                       default_q=DEFAULT_Q,
#                                       num_actions=len(options), rng=rng),
#                         options, 1.0, rng)
#         agentHRL = Agent(env, policyHRL, reward, rng)
#         try:
#             start = time()
#             agentHRL.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
#                            report=report)
#             end = time()
#             print("hrl: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                             end - start))
#         except KeyboardInterrupt:
#             logging.warning("interrupted! (hrl, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policyHRL
# del agentHRL
#
# for option in options:
#     option.policy.clear()
#
#
# a0 = HLAction([], [], [0], 0)
# a1 = HLAction([], [], [1], 1)
# a2 = HLAction([], [], [2], 2)
# a3 = HLAction([], [], [3], 3)
# a4 = HLAction([], [], [4], 4)
# a5 = HLAction([], [], [5], 5)
# a6 = HLAction([], [], [6], 6)
# a7 = HLAction([4], [4], [6, 7], 6)
# a8 = HLAction([5], [5], [6, 8], 6)
#
# plan0 = [a0]
# plan1 = [a1]
# plan2 = [a2]
# plan3 = [a3]
# plan4 = [a4]
# plan5 = [a5]
# plan6 = [a6]
# plan7 = [a4, a7]
# plan8 = [a5, a7]
# plan9 = [a4, a7, a5, a8]
# plan10 = [a0, a1, a2, a3]
# plan11 = [a4, a7, a5, a8, a0, a1, a2, a3]
# plan12 = [a4, a7, a5, a8, a4, a5, a0, a1, a2, a3]
#
# plans = [plan0, plan1, plan2, plan3, plan4, plan5, plan6, plan7, plan8, plan9,
#          plan10, plan11, plan12]
# plans = plans[START_TASK:END_TASK+1]
#
#
# with open("/tmp/office-shaped.csv", "w") as csvfile:
#     print("shaped: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("shaped: begin task {}".format(i + START_TASK))
#         reward = Shaped(ReachFacts(env, goal), plans[i], env.label, 1.0)
#         policy = EpsilonGreedy(alpha=1.0, gamma=1.0, epsilon=0.1,
#                                default_q=DEFAULT_Q, num_actions=4, rng=rng)
#         agent = Agent(env, policy, reward, rng)
#         try:
#             start = time()
#             agent.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
#                         report=report)
#             end = time()
#             print("shaped: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                                end - start))
#         except KeyboardInterrupt:
#             logging.warning("interrupted! (shaped, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policy
# del agent
#
#
# with open("/tmp/office-hshaped.csv", "w") as csvfile:
#     print("hshaped: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("hshaped: begin task {}".format(i + START_TASK))
#         reward = Shaped(ReachFacts(env, goal), plans[i], env.label, 1.0)
#         policy = HRL(EpsilonGreedy(alpha=0.1, gamma=1.0, epsilon=0.1,
#                                    default_q=DEFAULT_Q,
#                                    num_actions=len(options), rng=rng),
#                      options, 1.0, rng)
#
#         agent = Agent(env, policy, reward, rng)
#         try:
#             start = time()
#             agent.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
#                         report=report)
#             end = time()
#             print("hshaped: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                                end - start))
#         except KeyboardInterrupt:
#             logging.warning("interrupted! (hshaped, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policy
# del agent
#
# for option in options:
#     option.policy.clear()
#
#
# with open("tmp/office-seq.csv", "w") as csvfile:
#     print("seq: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("seq: begin task {}".format(i + START_TASK))
#         reward = ReachFacts(env, goal)
#         policySeq = SeqPlan(plans[i], options, rng)
#         agentSeq = Agent(env, policySeq, reward, rng)
#         try:
#             start = time()
#             agentSeq.train(steps=TOTAL_STEPS,
#                             steps_per_episode=EPISODE_LENGTH, report=report)
#             end = time()
#             print("seq: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                             end - start))
#         except KeyboardInterrupt:
#             logging.warning("Interrupted! (seq, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policySeq
# del agentSeq
#
# for option in options:
#     option.policy.clear()
#
#
# with open("tmp/office-monseq.csv", "w") as csvfile:
#     print("monseq: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("monseq: begin task {}".format(i + START_TASK))
#         reward = ReachFacts(env, goal)
#         policyMonSeq = MonSeq(plans[i], env.label, goal, options, rng)
#         agentMonSeq = Agent(env, policyMonSeq, reward, rng)
#         try:
#             start = time()
#             agentMonSeq.train(steps=TOTAL_STEPS,
#                               steps_per_episode=EPISODE_LENGTH, report=report)
#             end = time()
#             print("monseq: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                                end - start))
#         except KeyboardInterrupt:
#             logging.warning("Interrupted! (monseq, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policyMonSeq
# del agentMonSeq
#
# for option in options:
#     option.policy.clear()
#
#
# pops: List[Tuple[List[HLAction], List[Tuple[int, int]]]] = \
#     [(plan0, []), (plan1, []), (plan2, []), (plan3, []), (plan4, []),
#      (plan5, []), (plan6, []), (plan7, [(0, 1)]), (plan8, [(0, 1)]),
#      (plan9, [(0, 1), (2, 3)]), (plan10, []), (plan11, [(0, 1), (2, 3)]),
#      (plan12, [(0, 1), (2, 3), (1, 4), (3, 5)])]
# pops = pops[START_TASK:END_TASK+1]
#
# with open("/tmp/office-pop.csv", "w") as csvfile:
#     print("pop: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("pop: begin task {}".format(i + START_TASK))
#         reward = ReachFacts(env, goal)
#         policyPOP = POPlan(pops[i],
#                            EpsilonGreedy(alpha=0.1, gamma=1.0, epsilon=0.1,
#                                          default_q=DEFAULT_Q,
#                                          num_actions=len(options), rng=rng),
#                            options, 1.0, rng)
#         agentPOP = Agent(env, policyPOP, reward, rng)
#         try:
#             start = time()
#             agentPOP.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
#                            report=report)
#             end = time()
#             print("pop: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                             end - start))
#         except KeyboardInterrupt:
#             logging.warning("interrupted! (pop, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policyPOP
# del agentPOP
#
# for option in options:
#     option.policy.clear()
#
# with open("tmp/office-monpop.csv", "w") as csvfile:
#     print("monpop: begin experiment")
#     report = SequenceReport(csvfile, LOG_STEP, states, 1000, TRIALS)
#     for i, (goal, steps) in enumerate(tasks):
#         print("monpop: begin task {}".format(i + START_TASK))
#         reward = ReachFacts(env, goal)
#         policyMon = MonPOP(pops[i], env.label, goal,
#                            EpsilonGreedy(alpha=0.9, gamma=1.0, epsilon=0.75,
#                                          default_q=DEFAULT_Q,
#                                          num_actions=len(options), rng=rng),
#                            options, 1.0, rng)
#         agentMon = Agent(env, policyMon, reward, rng)
#         try:
#             start = time()
#             agentMon.train(steps=TOTAL_STEPS, steps_per_episode=EPISODE_LENGTH,
#                            report=report)
#             end = time()
#             print("monpop: Ran task {} for {} seconds.".format(i + START_TASK,
#                                                                end - start))
#         except KeyboardInterrupt:
#             logging.warning("interrupted! (monpop, task %s)", i + START_TASK)
#
#         report.increment(TOTAL_STEPS)
#
# del policyMon
# del agentMon
#
# for option in options:
#     option.policy.clear()
