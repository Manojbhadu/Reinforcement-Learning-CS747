# import sys
# import argparse
import random
import numpy as np

def eps_greedy(true_mean, horizon,eps,seed):
    random.seed(seed)
    arms = len(true_mean)
    emp_mean = np.zeros(arms,dtype = float)
    count = np.zeros(arms,dtype = float)
    reward = np.zeros(arms,dtype = float)
    for i in range(arms):
        pull = random.randint(0,arms-1)
        count[pull]+=1
        p = random.random()
        if p < true_mean[pull]:
            reward[pull]+=1

        emp_mean = np.divide(reward, count, out=np.zeros_like(reward), where=count!=0)
        

    for i in range((horizon-arms)):
        pull_decision = random.random()
        if pull_decision <= eps :
            pull = random.randint(0,arms-1)
            count[pull]+=1
            p = random.random()
            if p < true_mean[pull]:
                reward[pull]+=1

            emp_mean = np.divide(reward, count, out=np.zeros_like(reward), where=count!=0)

        if pull_decision > eps:
            pull = np.argmax(emp_mean)
            count[pull]+=1
            p = random.random()
            if p < true_mean[pull]:
                reward[pull]+=1

            emp_mean = np.divide(reward, count, out=np.zeros_like(reward), where=count!=0)

    total_reward = np.sum(reward)
    max_reward = horizon*(np.max(true_mean))

    regret = max_reward - total_reward

    return regret


def UCB(true_mean, horizon,seed):
    random.seed(seed)
    arms = len(true_mean)
    count_t = np.zeros(arms,dtype = float)
    reward = np.zeros(arms,dtype = float)
    emp_mean = np.zeros(arms,dtype = float)
    ucb_t = np.zeros(arms,dtype = float)

    for i in range(arms):
        pull = i
        count_t[pull]+=1
        p = random.random()
        if p < true_mean[pull]:
            reward[pull]+=1
    emp_mean = reward/count_t 
    for i in range(arms,arms*2):
        pull = random.randint(0,arms-1)
        count_t[pull]+=1
        p = random.random()
        if p < true_mean[pull]:
            reward[pull]+=1
    emp_mean = reward/count_t

    for i in range(2*arms,horizon):
        ucb_t = emp_mean + (np.sqrt(2*np.log(i))/np.sqrt(count_t))
        pull = np.argmax(ucb_t)
        count_t[pull]+=1
        p = random.random()
        if p < true_mean[pull]:
            reward[pull]+=1

        emp_mean[pull] = reward[pull]/count_t[pull]

    total_reward = np.sum(reward)
    max_reward = horizon*(np.max(true_mean))
    regret = max_reward - total_reward

    return regret


def KLUCB(true_mean, horizon,seed, precision = 1e-6):
    random.seed(seed)
    arms = len(true_mean)
    count_t = np.zeros(arms,dtype = float)
    reward = np.zeros(arms,dtype = float)
    emp_mean = np.zeros(arms,dtype = float)
    kl_ucb_t = np.zeros(arms,dtype = float)
    def kl(p,q):
        thr = 1e-15
        x = min(max(p, thr), 1 - thr)
        y = min(max(q, thr), 1 - thr)
        return x * np.log(x/y) + (1-x)*np.log((1-x) / (1-y))

    def qmax(p_hat, t, uat, c=0,precision = precision):
        ub = (np.log(t)+c*np.log(np.log(t)))/uat
        lq = p_hat
        rq = 1.0
        q_opt = (lq+rq)/2
        kld = kl(p_hat,q_opt)
        while kld > ub or ub - kld > precision :
            if kld > ub :
                rq = q_opt 
            else:
                lq = q_opt

            q_opt = (rq+lq)/2
            kld = kl(p_hat, q_opt)
            if (1-q_opt < precision):
                break
        return q_opt
    for i in range(arms):
        pull = i
        count_t[pull]+=1
        p = random.random()
        if p < true_mean[pull]:
            reward[pull]+=1
    emp_mean  = reward/count_t
    for i in range(arms):
        kl_ucb_t[i] =  qmax(emp_mean[i], arms,count_t[i])

    for t in range(arms, horizon):
        pull = np.argmax(kl_ucb_t)
        count_t[pull] +=1
        p = random.random()
        if p < true_mean[pull]:
            reward[pull]+=1

        emp_mean[pull] = reward[pull]/count_t[pull]
        for i in range(arms):
            kl_ucb_t[i] =  qmax(emp_mean[i], t+1,count_t[i])

    total_reward = np.sum(reward)
    max_reward = horizon*(np.max(true_mean))
    regret = max_reward - total_reward

    return regret


    
def thompson_sampling(true_mean, horizon,seed):
    random.seed(seed)
    arms = len(true_mean)
    suc_t = np.zeros(arms)
    #fail_t = np.zeros(arms)
    count_t = np.zeros(arms)
    beta_t = np.zeros(arms)
    
    for t in range(horizon):
        for i in range(arms):
            beta_t[i] = random.betavariate(suc_t[i]+1, count_t[i]-suc_t[i]+1)
        pull = np.argmax(beta_t)
        count_t[pull]+=1
        p = random.random()
        if p < true_mean[pull]:
            suc_t[pull]+=1

    total_reward = np.sum(suc_t)
    max_reward = horizon*(np.max(true_mean))
    regret = max_reward - total_reward

    return regret




def thompson_sampling_with_hint(true_mean, horizon, seed, additional_info):
    random.seed(seed)
    arms = len(true_mean)
    suc_t = np.zeros(arms)
    #fail_t = np.zeros(arms)
    count_t = np.zeros(arms)
    beta_t = np.zeros(arms)
    emp_mean = np.zeros(arms)
    diff = additional_info[-1]-additional_info[-2]
    for i in range(arms):
            beta_t[i] = random.betavariate(suc_t[i]+1, count_t[i]-suc_t[i]+1)
    pull = np.argmax(beta_t)
    count_t[pull]+=1
    p = random.random()
    if p < true_mean[pull]:
        suc_t[pull]+=1

    emp_mean[pull] = suc_t[pull]/count_t[pull]

    emp_max = np.max(emp_mean)
    
    for t in range(1,horizon):
        if emp_max >= additional_info[-1] or additional_info[-1]-emp_max < diff:
            pull = np.argmax(emp_mean)
            count_t[pull]+=1
            p = random.random()
            if p < true_mean[pull]:
                suc_t[pull]+=1

            emp_mean[pull] = suc_t[pull]/count_t[pull]

        else:
            for i in range(arms):
                beta_t[i] = random.betavariate(suc_t[i]+1, count_t[i]-suc_t[i]+1)
            pull = np.argmax(beta_t)
            count_t[pull]+=1
            p = random.random()
            if p < true_mean[pull]:
                suc_t[pull]+=1

            emp_mean[pull] = suc_t[pull]/count_t[pull]


        emp_max = np.max(emp_mean)


    total_reward = np.sum(suc_t)
    max_reward = horizon*(np.max(true_mean))
    regret = max_reward - total_reward

    return regret






