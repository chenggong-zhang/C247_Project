import numpy as np
import pickle
import matplotlib.pyplot as plt


random_policy_result = np.loadtxt('./random_policy/Market_Environments/random_policy_results.txt')

with open('./DDPG_model/results/data_and_graphs/Mak_DDPG.pkl', 'rb') as f:
    DDPG_result = pickle.load(f)

with open('./SAC_model/results/data_and_graphs/Mak_SAC.pkl', 'rb') as f:
    SAC_result = pickle.load(f)

with open('./TD3_model/results/numerical_results/TD3.pkl', 'rb') as f:
    TD3_result = pickle.load(f)

DDPG = np.zeros((5, 21))
SAC = np.zeros((5, 21))
TD3 = np.zeros((5, 20))
indices = np.array(np.linspace(0, 200, 21), dtype=int)
for i in range(5):
    DDPG[i,:] = np.array(DDPG_result['DDPG'][i][1])
    SAC[i,:] = np.array(SAC_result['SAC'][i][1])[indices]
    TD3[i,:] = np.array(TD3_result['TD3'][i][1])

mean_results_DDPG = np.mean(DDPG, axis=0)
std_results_DDPG = np.std(DDPG, axis=0)
mean_minus_std_DDPG = mean_results_DDPG - std_results_DDPG
mean_plus_std_DDPG = mean_results_DDPG + std_results_DDPG

mean_results_SAC = np.mean(SAC, axis=0)
std_results_SAC = np.std(SAC, axis=0)
mean_minus_std_SAC = mean_results_SAC - std_results_SAC
mean_plus_std_SAC = mean_results_SAC + std_results_SAC

mean_results_TD3 = np.mean(TD3, axis=0)
std_results_TD3 = np.std(TD3, axis=0)
mean_minus_std_TD3 = mean_results_TD3 - std_results_TD3
mean_plus_std_TD3 = mean_results_TD3 + std_results_TD3


plt.figure(dpi=600)
ax = plt.gca()
ax.plot(indices[:-1], mean_results_DDPG[:-1], label='DDPG', color='#B51515')
ax.plot(indices[:-1], mean_plus_std_DDPG[:-1], color='#B51515', alpha=0.1)
ax.plot(indices[:-1], mean_minus_std_DDPG[:-1], color='#B51515', alpha=0.1)
ax.fill_between(indices[:-1], y1=mean_minus_std_DDPG[:-1], y2=mean_plus_std_DDPG[:-1],
                alpha=0.1, color='#B51515')

ax.plot(indices[:-1], mean_results_SAC[:-1], label='SAC', color='#1C2833')
ax.plot(indices[:-1], mean_plus_std_SAC[:-1], color='#1C2833', alpha=0.1)
ax.plot(indices[:-1], mean_minus_std_SAC[:-1], color='#1C2833', alpha=0.1)
ax.fill_between(indices[:-1], y1=mean_minus_std_SAC[:-1], y2=mean_plus_std_SAC[:-1],
                alpha=0.1, color='#1C2833')

ax.plot(indices[:-1], mean_results_TD3, label='TD3', color='#5B2C6F')
ax.plot(indices[:-1], mean_plus_std_TD3, color='#5B2C6F', alpha=0.1)
ax.plot(indices[:-1], mean_minus_std_TD3, color='#5B2C6F', alpha=0.1)
ax.fill_between(indices[:-1], y1=mean_minus_std_TD3, y2=mean_plus_std_TD3, alpha=0.1, color='#5B2C6F')
ax.plot(indices[:-1], np.mean(random_policy_result)*np.ones(20), label='random policy', linestyle='dashed')
plt.xlabel('Episodes')
plt.ylabel('Profit in US Dollar')
plt.xlim(0, 201)
plt.ylim(-40, 81)
plt.legend()
plt.show()