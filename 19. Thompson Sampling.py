# Thompson Sampling
'''
--- Thompson Sampling is a Probabilistic Algorithm, while Upper Confidence Bound was a 
    Deterministic Algorithm.
--- UCB requires the adjustment in the algorithm after each & every round; while Thopsom 
    Sampling can accomodate the delay in feedback. (So, Thompson Sampling can be considered
    as a Faster Algorithm)
--- Thompson Sampling have a better empirical evidence. 
'''


'''
Before approaching the algorithm, it is better to determine the BETA Distribution

'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg



# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Understanding the Process & approch for Thomspon Sampling
mbp_approach = mpimg.imread('Multiarm Bandit Problem Approach.PNG')
plt.imshow(mbp_approach)

#Understandingthe Bayesian Interface
by_int = mpimg.imread('Bayesian Interface.PNG')
plt.imshow(by_int)

#Understanding the Thompson Sampling Algorithm
tm_sm_algorithm = mpimg.imread('Thompson Sampling Algorithm.PNG')
plt.imshow(tm_sm_algorithm)

'''
After Step 1, 
Step 2 considers Bayesian Inference inorder to complete Step 2
'''


# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d # Step 1 - number of times ad 'i' got reward 1 upto round 'n'
numbers_of_rewards_0 = [0] * d # Step 1 - number of times ad 'i' got reward 0 upto round 'n'
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) # Gives random draws for Beta Distribution
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward

# Visualising the results - Histogram
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

'''
So, it can be seen that Ad number 5, was the most clicked.
As compared with UCB, the thompson Sampling is approaching the results quickly & is giving
far more desirable results.
'''

'''