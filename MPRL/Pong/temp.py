import matplotlib.pyplot as plt
episode_rewards = [1,2,3,4,5,6,7,8]
partial=[None,2,None,4,None,None,None,None]

plt.plot(episode_rewards,"bs")
plt.plot(partial, "ro")
plt.ylabel('reward')
plt.xlabel('episode')
plt.show()