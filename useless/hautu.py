import matplotlib.pyplot as plt

def plot_training_results(timesteps, rewards, success_rates):
    plt.figure(figsize=(10, 4))

    # 平均奖励折线图
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, rewards, marker='o')
    plt.title("Average Reward vs Training Timesteps")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Reward")
    plt.grid(True)

    # 成功率折线图
    plt.subplot(1, 2, 2)
    plt.plot(timesteps, [s * 100 for s in success_rates], marker='o')
    plt.title("Success Rate vs Training Timesteps")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 示例调用
timesteps = [1000, 3000, 5000, 10000, 20000]
avg_rewards = [-3.2, -2.1, -1.5, -0.8, -0.5]
success_rates = [0.1, 0.4, 0.65, 0.85, 0.9]

plot_training_results(timesteps, avg_rewards, success_rates)
