import random
import sys
from time import time


def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return best[0]#random.choice(best)


def get_actions(s,A,T):
    return [a for a in A if (s,a) in T]

def run_qlearning(S,A,T,R0,R1,DONE,s0,alpha):
    # HYPER-PARAMETERS
    lr      = 1.0
    epsilon = 0.0
    gamma   = 1.0
    q_init  = 0.0
    training_steps = 10000000
    ns_episode = 100


    #if show:
    #    print("N_step (M)\tTrain (100*rw/st) \t Test (100*rw/st) \t Test (rw/ep)")

    # Running Q-Learning
    step = 0
    Q = {}
    while step < training_steps:
        total_r0 = 0
        total_r1 = 0
        total_delta = 0
        s = s0
        if s not in Q: Q[s] = dict([(a,q_init) for a in get_actions(s,A,T)])
        for _ in range(ns_episode):
            # Selecting and executing the action
            if random.random() < epsilon: a = random.choice(get_actions(s,A,T))
            else:
                a = get_best_action(Q,s,get_actions(s,A,T),q_init)
            sn = T[(s,a)]
            r  = R0[(s,a)] + alpha*R1[(s,a)]
            done = sn in DONE

            # Updating the q-values
            if done: delta = r - Q[s][a]
            else:    delta = r + gamma*get_qmax(Q,sn,get_actions(sn,A,T),q_init) - Q[s][a]
            Q[s][a] += lr*delta


            # moving to the next state
            total_r0 += R0[(s,a)]
            total_r1 += R1[(s,a)]
            total_delta += delta
            step += 1
            """
            if step%testing_freq == 0:
                test_reward, test_episodes  = test_policy(env_args, testing_steps, ns_episode, Q, 0)
                train_performance = 100*reward_total/testing_freq
                test_performance  = 100*test_reward/testing_steps
                test_rew_ep       = test_reward/test_episodes if test_episodes > 0 else 0
                results.append((step/1000000, train_performance, test_performance, test_rew_ep))
                if show: print(step/1000000, ("\t\t%0.3f"%train_performance), "\t\t\t", test_performance, "\t\t\t", test_rew_ep)
                reward_total = 0
            """

            if done or not(step < training_steps):
                break
            s = sn
        #print(total_delta, "\t", total_r0, "\t", total_r1)
        if total_delta == 0:
            break

    policy = {}
    for s in S:
        policy[s] = get_best_action(Q,s,get_actions(s,A,T),q_init)

    return policy



def clip(a,a_min,a_max):
    return max([a_min,min([a,a_max])])


def move_towards(x1,y1,x2,y2):
    if x1 < x2:
        return 'd'
    if x1 > x2:
        return 'a'
    if y1 < y2:
        return 's'
    if y1 > y2:
        return 'w'
    return 'e'


def get_model():
    """
    This method returns a model of the environment.
    We use the model to compute optimal policies using value iteration.
    The optimal policies are used to set the average reward per step of each task to 1.
    """
    # Key: 0,1,2,3 (agent 0, agent 1, pos 2, pos 3)
    S = [(x0,y0,x1,y1,k,w0,w1,h0,h1,b0,b1) for x0 in range(8) for y0 in range(8) for x1 in range(8) for y1 in range(8) for k in range(4) for w0 in range(2) for w1 in range(2) for h0 in range(2) for h1 in range(2) for b0 in range(2) for b1 in range(2)] # States
    A = ['a','s','d','w','e']

    T = {}
    R0 = {}
    R1 = {}
    DONE = set()

    # Map locations
    locations = {}
    locations[(0,0)] = 'E'
    locations[(7,7)] = 'E'
    locations[(2,0)] = 'K'
    locations[(6,6)] = 'K'
    locations[(7,0)] = 'H'
    locations[(5,4)] = 'W'
    locations[(4,5)] = 'F'

    for s in S:
        for a0 in A:
            x0,y0,x1,y1,k,w0,w1,h0,h1,b0,b1 = s
            if a0 == 'e' and ((x0,y0) not in locations or locations[(x0,y0)] == 'E'):
                continue
            # First agent move
            r0 = 0
            if not(b0 == 1 and (x0,y0) in locations and locations[(x0,y0)] == 'E'):
                if a0 == 'a': x0 -= 1
                if a0 == 'd': x0 += 1
                if a0 == 'w': y0 -= 1
                if a0 == 's': y0 += 1
                x0 = clip(x0,0,7)
                y0 = clip(y0,0,7)
                # wood
                if a0 == 'e':
                    if locations[(x0,y0)] == 'W':
                        w0 = 1
                    if locations[(x0,y0)] == 'H':
                        h0 = 1
                    if locations[(x0,y0)] == 'K':
                        # Returning key
                        if k == 0:
                            if (x0,y0) == (2,0):
                                k = 2
                            else:
                                k = 3
                        # Picking up key
                        elif k == 2 and (x0,y0) == (2,0):
                            k = 0
                        elif k == 3 and (x0,y0) == (6,6):
                            k = 0
                    if locations[(x0,y0)] == 'F':
                        if w0 == 1 and h0 == 1 and k == 0:
                            b0 = 1
                r0 = -1
            # Second agent
            r1 = 0
            if not(b1 == 1 and (x1,y1) in locations and locations[(x1,y1)] == 'E'):
                # Selecting an action
                if k == 2:
                    a1 = move_towards(x1,y1,2,0)
                elif h1 == 0:
                    a1 = move_towards(x1,y1,7,0)
                elif w1 == 0:
                    a1 = move_towards(x1,y1,5,4)
                elif b1 == 0 and k == 1:
                    a1 = move_towards(x1,y1,4,5)
                elif b1 == 1 and k == 1:
                    a1 = move_towards(x1,y1,6,6)
                elif b1 == 1:
                    a1 = move_towards(x1,y1,7,7)
                else:
                    a1 = move_towards(x1,y1,6,6)

                if a1 == 'a': x1 -= 1
                if a1 == 'd': x1 += 1
                if a1 == 'w': y1 -= 1
                if a1 == 's': y1 += 1
                x1 = clip(x1,0,7)
                y1 = clip(y1,0,7)
                # wood
                if a1 == 'e':
                    if locations[(x1,y1)] == 'W':
                        w1 = 1
                    if locations[(x1,y1)] == 'H':
                        h1 = 1
                    if locations[(x1,y1)] == 'K':
                        # Returning key
                        if k == 1:
                            if (x1,y1) == (2,0):
                                k = 2
                            else:
                                k = 3
                        # Picking up key
                        elif k == 2 and (x1,y1) == (2,0):
                            k = 1
                        elif k == 3 and (x1,y1) == (6,6):
                            k = 1
                    if locations[(x1,y1)] == 'F':
                        if w1 == 1 and h1 == 1 and k == 1:
                            b1 = 1
                r1 = -1
            # Next state
            s2 = x0,y0,x1,y1,k,w0,w1,h0,h1,b0,b1
            T[(s,a0)] = s2
            s2_done0 = b0 == 1 and (x0,y0) in locations and locations[(x0,y0)] == 'E'
            s2_done1 = b1 == 1 and (x1,y1) in locations and locations[(x1,y1)] == 'E'
            if s2_done0 and s2_done1:
                DONE.add(s2)
            elif s2_done0 and k ==0:
                DONE.add(s2)
                r1 = -100
            R0[(s,a0)] = r0
            R1[(s,a0)] = r1

   # print(len(S))

    s0 = 0,0,0,0,2,0,0,0,0,0,0
    return S,A,T,R0,R1,DONE,s0


def show_map(s,a0):
    locations = {}
    locations[(0,0)] = 'E'
    locations[(7,7)] = 'E'
    locations[(2,0)] = 'K'
    locations[(6,6)] = 'K'
    locations[(7,0)] = 'H'
    locations[(5,4)] = 'W'
    locations[(4,5)] = 'F'

    x0,y0,x1,y1,k,w0,w1,h0,h1,b0,b1 = s
    for y in range(8):
        for x in range(8):
            if (x,y) == (x0,y0):
                print('0',end='')
            elif (x,y) == (x1,y1):
                print('1',end='')
            elif (x,y) in locations:
                print(locations[(x,y)],end='')
            else:
                print(" ",end='')
        print()
    if a0 is not None:
        print("Reward:", R0[(s,a0)], R1[(s,a0)])
    print("Key:", k)
    print("Wood:", w0, w1)
    print("Hammer:", h0, h1)
    print("Box:", b0, b1)


def test_policy(S,A,T,R0,R1,DONE,s0,policy):
    s = s0
    total_r0 = 0
    total_r1 = 0
    #show_map(s,None)
    for _ in range(100):
        a0 = policy[s]
        total_r0 += R0[(s,a0)]
        total_r1 += R1[(s,a0)]
        #show_map(s,a0)
        s = T[(s,a0)]
        if s in DONE:
            break

    print("Total Reward of First Agent:", total_r0)
    print("Total Reward of Second Agent:", total_r1)


if __name__ == '__main__':
    start = time()
    alpha = float(sys.argv[1]) # options: [0.0, 0.5, 1.5]
    #alpha = 0
    S,A,T,R0,R1,DONE,s0 = get_model()
    policy = run_qlearning(S,A,T,R0,R1,DONE,s0,alpha)
    test_policy(S,A,T,R0,R1,DONE,s0,policy)
    end = time()
    print("Total Time:", end - start)
