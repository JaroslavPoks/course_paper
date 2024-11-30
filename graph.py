import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.DataFrame(np.load('error.npy'), columns=['E1 error', 'E2 error', 'E3 error', 'E4 error'])

mean_4 = round(np.mean(data['E4 error']), 4)
mean_3 = round(np.mean(data['E3 error']), 4)
mean_2 = round(np.mean(data['E2 error']), 4)
mean_1 = round(np.mean(data['E1 error']), 4)

std_4  = round(np.std(data['E4 error']), 4)
std_3  = round(np.std(data['E3 error']), 4)
std_2  = round(np.std(data['E2 error']), 4)
std_1  = round(np.std(data['E1 error']), 4)

# print(mean_4, std_4)
# dist = sns.displot(data['E4 error'], kind='kde', color='blue', height=5, aspect=1.5)
# dist.fig.suptitle(f'Mean: {mean_4}, Std: {std_4}')
# plt.savefig('density.png')
dist4 = sns.displot(data['E4 error'], bins=30, kind='hist', color='blue', height=5, aspect=1.5)
dist4.fig.suptitle(f'Mean: {mean_4}, Std: {std_4}')
plt.savefig('hist4.png')

dist3 = sns.displot(data['E3 error'], bins=30, kind='hist', color='blue', height=5, aspect=1.5)
dist3.fig.suptitle(f'Mean: {mean_3}, Std: {std_3}')
plt.savefig('hist3.png')

dist2 = sns.displot(data['E2 error'], bins=30, kind='hist', color='blue', height=5, aspect=1.5)
dist2.fig.suptitle(f'Mean: {mean_2}, Std: {std_2}')
plt.savefig('hist2.png')

dist1 = sns.displot(data['E1 error'], bins=30, kind='hist', color='blue', height=5, aspect=1.5)
dist1.fig.suptitle(f'Mean: {mean_1}, Std: {std_1}')
# plt.xlim(0, 10)
plt.savefig('hist1.png')
