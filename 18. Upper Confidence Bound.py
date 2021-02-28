# Upper Confidence Bound (UCB)

##########          REINFORCEMENT LEARNING          #############
'''
Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is
used to solve interacting problems where the data observed up to time t is considered
to decide which action to take at time t + 1. It is also used for Artificial Intelligence
when training machines to perform tasks such as walking. Desired outcomes provide
the AI with reward, undesired with punishment. Machines learn through trial and error.

Types:::
    1. Upper Confidence Bound(UCB)
    2. Thompson Sampling
'''
'''
    Multi Armed Bandit Problem::
        Its main objective is to maximize reward over a given number of time steps
    -- It maximize the rewards through repeated actions
    -- It is a ratio of """sum of rewards when X step is taken prior to time t""" to
        """number of X steps taken prior to time t"""
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
 
#Implementing Random Selection (ONLY FOR REFERENCE)
import random
Nr = 10000
dr = 10
ads_selected_r = []
total_reward_r = 0

for m in range(0,Nr):
    adr = random.randrange(dr)
    ads_selected_r.append(adr)
    rewards_r = dataset.values[m, adr]
    total_reward_r = total_reward_r + rewards_r

'''
When comparing """ads_selected_r""" & """dataset""", it can be noticed that the
random model has many imperfect predictions.
'''


'''********* MATHS behind UCB **************'''

import matplotlib.image as mpimg
img = mpimg.imread('UCB Steps_01.PNG')
plt.imshow(img)

# Implementing UCB
import math # inorder to use Squareroot function
N = 10000 # total number of round(10000 is max users)
d = 10 # max ads in for each users
ads_selected = [] # for step 3
numbers_of_selections = [0] * d # Ni(n) = number of times ad 'i' was selected upto round 'n'
sums_of_rewards = [0] * d # Ri(n) = sum of rewards of ad 'i' upto round 'n'
total_reward = 0 # total reward at the start of loop
for n in range(0, N): # For STEP 2 (N = total number of round)
    ad = 0 #step 3
    max_upper_bound = 0 #step 3
    for i in range(0, d): # for 'i' in range upto round 'n'
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i] # average reward of ad 'i' up tp round n
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # Delta for confidence interval
            upper_bound = average_reward + delta_i # Upper confidence bound
        else:
            upper_bound = 1e400 # 10 at the power of 400 (a very large number)
        if upper_bound > max_upper_bound: #step 3
            max_upper_bound = upper_bound #step 3
            ad = i #step 3
    ads_selected.append(ad) # used to update the result after each round
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1 # updation till next round
    reward = dataset.values[n, ad] # Rewards of nth loop
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward # sum of rewards
    total_reward = total_reward + reward # total rewards after the end of loop

'''
the output '''ads_selected''' shows that the ads selected by the algorithm.
As can be seen from the '''ads_selected''' dataset, the ad no.4 is repeated the most.
Also, as can be seen, the ad no.4 is selected the most for the last few instances,
this is because, the models confidence level is converged, and it will not change
after the long list of observations.
'''


# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

'''
From the end of code, it can be easily concluded from the histogram,
the BEST ADVERTISEMENT IS AD NO.4
'''