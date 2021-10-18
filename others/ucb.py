import numpy as np


class MAB:
    def __init__(self, alpha, epochs, num_arms):
        self.alpha = alpha
        self.epochs = epochs
        self.num_arms = num_arms
        self.rewards = np.zeros(self.num_arms)
        self.trials = np.zeros(self.num_arms)
        self.p_arms = np.random.uniform(low=0, high=1, size=self.num_arms)

    def alpha_greedy(self):
        if np.random.random() < self.alpha:
            arm = np.random.randint(low=0, high=self.num_arms)
        else:
            arm = np.argmax(self.p_arms)
        reward = np.random.binomial(n=1, p=self.p_arms[arm])
        return arm, reward

    def __call__(self):
        total_reward = 0
        for i in range(self.epochs):
            arm, reward = self.alpha_greedy()
            self.trials[arm] += 1
            self.rewards[arm] += reward
            self.p_arms[arm] = self.rewards[arm]/self.trials[arm]
            total_reward += reward
        return total_reward


class UCB:
    def __init__(self, epochs, num_arms):
        self.epochs = epochs
        self.num_arms = num_arms
        self.rewards = np.zeros(self.num_arms)
        self.trials = np.zeros(self.num_arms)
        self.p_arms = np.random.uniform(low=0, high=1, size=self.num_arms)

    def calculate_delta(self, epoch, arm):
        if self.trials[arm] == 0:
            return 1
        else:
            return np.sqrt(2 * np.log(epoch)/self.trials[arm])

    def ucb(self, epoch):
        upper_bound_probs = [self.p_arms[i] + self.calculate_delta(epoch, i) for i in range(self.num_arms)]
        arm = np.argmax(upper_bound_probs)
        reward = np.random.binomial(n=1, p=self.p_arms[arm])
        return arm, reward

    def __call__(self):
        total_reward = 0
        for epoch in range(1, self.epochs+1):
            arm, reward = self.ucb(epoch)
            self.trials[arm] += 1
            self.rewards[arm] += reward
            self.p_arms[arm] = self.rewards[arm]/self.trials[arm]
            total_reward += reward
        return total_reward


if __name__ == '__main__':
    # mab = MAB(0.2, 100, 10)
    # print(mab())
    print(UCB(100, 10)())
