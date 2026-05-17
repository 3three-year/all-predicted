import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, state_dim))
        self.action_memory = np.zeros((self.mem_size,))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, state_dim))
        self.terminal_memory = np.zeros((self.mem_size,), dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size
        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)
        return (self.state_memory[batch],
                self.action_memory[batch],
                self.reward_memory[batch],
                self.next_state_memory[batch],
                self.terminal_memory[batch])

    def ready(self):
        return self.mem_cnt > self.batch_size

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_size, batch_size, alpha=0.4):
        super().__init__(state_dim, action_dim, max_size, batch_size)
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.alpha = alpha
        self.max_priority = 1.0

    def store_transition(self, state, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size

        # 预分配内存避免重复创建数组
        if self.state_memory is None:
            self.state_memory = np.zeros((self.mem_size, *state.shape))
            self.next_state_memory = np.zeros((self.mem_size, *state_.shape))

        self.state_memory[mem_idx] = state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_
        self.terminal_memory[mem_idx] = done
        self.priorities[mem_idx] = self.max_priority

        self.mem_cnt += 1

    def sample_buffer(self):
        mem_len = min(self.mem_size, self.mem_cnt)
        if mem_len < self.batch_size:
            indices = np.arange(mem_len)
        else:
            probs = self.priorities[:mem_len] ** self.alpha
            probs_sum = probs.sum()
            if probs_sum < 1e-6:
                probs = np.ones(mem_len) / mem_len
            else:
                probs /= probs_sum
            indices = np.random.choice(mem_len, self.batch_size, p=probs, replace=False)
        return indices, (self.state_memory[indices],
                         self.action_memory[indices],
                         self.reward_memory[indices],
                         self.next_state_memory[indices],
                         self.terminal_memory[indices])

    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities) + 1e-5
        for idx, prio in zip(indices, priorities):
            self.priorities[idx % self.mem_size] = prio
        self.max_priority = max(self.max_priority, np.max(priorities))