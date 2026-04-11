# traffic_env.py  — improved reward function for faster, better learning

import gymnasium as gym
import numpy as np
import headless_sim as sim

prev_green:any

class TrafficEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        # [q_right, q_down, q_left, q_up, current_green, phase_time]
        self.observation_space = gym.spaces.Box(
            low=np.zeros(10),
            high=np.array([200]*4 + [300]*4 + [3, 300]),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self._phase_time = 0
        self._prev_crossed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sim.reset_globals()
        self._phase_time = 0
        self._prev_crossed = 0
        return self._obs(), {}

    def step(self, action):
            

        action = int(np.array(action).flatten()[0])
        prev_green = sim.currentGreen

        # Enforce minimum green time
        MIN_GREEN = 2

        if action != prev_green and self._phase_time < MIN_GREEN:
            action = prev_green
            
        queue, total_wait, done = sim.step_tick(action)
        self._phase_time = self._phase_time + 1 if action == prev_green else 0

            # ── reward ────────────────────────────────────────────────────────────
            # 1. Primary: reward every vehicle that crossed this tick
            # vehicles passed
        current_crossed = self._total_crossed()
        crossed_now = current_crossed - self._prev_crossed
        self._prev_crossed = current_crossed

        #  FINAL REWARD

        # 1. Throughput reward
        reward = crossed_now * 30.0

        # 2. Penalize total congestion
        reward -= sum(queue) * 0.1

        # 3. Penalize ignoring other lanes
        for i, q in enumerate(queue):
            if i != action:
                reward -= q * 0.03

        # 4. Switching penalty
        if action != prev_green:
            reward -= 10.0
        else:
            reward += 2.0    

        obs = self._obs()
        return obs, reward, done, False, {'queue': queue, 'crossed': current_crossed}

    def _obs(self):
        q = sim.get_queue_lengths()

        #  NEW: average waiting time per direction
        avg_wait = []
        for i in range(4):
            d = sim.directionNumbers[i]
            waits = []
            for lane in range(3):
                for v in sim.vehicles[d][lane]:
                    if not v['crossed']:
                        waits.append(v['wait_ticks'])
            avg_wait.append(np.mean(waits) if waits else 0)

        return np.array(q + avg_wait + [sim.currentGreen, self._phase_time], dtype=np.float32)

    def _total_crossed(self):
        return sum(sim.vehicles[sim.directionNumbers[i]]['crossed'] for i in range(4))


if __name__ == "__main__":
    env = TrafficEnv()
    obs, _ = env.reset()
    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)
    print("First obs:", obs)

    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
    print(f"Episode done. Reward={total_reward:.1f}  Crossed={info['crossed']}")
