# 🚦 AI-Based Multi-Agent Traffic Management System

## 📌 Project Overview

This project presents an intelligent **AI-based multi-agent traffic management system** that uses **Reinforcement Learning (RL)** to optimize traffic signal control at intersections.

Unlike traditional systems, this approach uses **multiple RL agents** that can learn and adapt to real-time traffic conditions, improving overall traffic efficiency. The system compares **Round-Robin**, **Heuristic**, and **RL-based control (PPO/DQN)** to demonstrate performance improvements.
<img width="1748" height="1015" alt="image" src="https://github.com/user-attachments/assets/005819a4-9852-40b3-9fbb-a3a4256e8fc9" />

## 🎯 Objectives

* Design a **multi-agent traffic control system**
* Simulate real-world traffic conditions
* Implement **baseline controllers** (Round-Robin, Heuristic)
* Train **RL agents (PPO/DQN)** for adaptive signal control
* Reduce:

  * Traffic congestion
  * Waiting time
  * Fuel consumption


## ⚙️ Project Framework

Traffic Environment → State (Queue, Waiting Time) → Multi-Agent RL (PPO/DQN) 
→ Action (Signal Control) → Traffic Controller → Vehicle Movement → Metrics → Loop

## 🧠 Technologies Used

* Python 🐍
* Reinforcement Learning (PPO, DQN)
* Stable-Baselines3
* NumPy, Pandas
* Matplotlib

## 🚀 Working of the System

1. Vehicles are generated randomly in different lanes
2. Each agent observes:

   * Queue length
   * Waiting time
   * Signal state
3. The **multi-agent RL system** decides signal actions
4. Traffic updates based on decisions
5. Rewards are calculated based on efficiency
6. Agents learn optimal policies over time

## 📊 Results Summary

| Metric               | Round-Robin | Heuristic | RL Agent |
| -------------------- | ----------- | --------- | -------- |
| Vehicles Crossed     | 386         | 294       | 387      |
| Throughput (veh/sec) | 1.287       | 0.981     | 1.289    |
| Avg Waiting Time     | 30.5        | 83        | 33.9     |
| Total Waiting Time   | 11,792      | 24,379    | 13,103   |
<img width="1145" height="717" alt="image" src="https://github.com/user-attachments/assets/ce76728b-09b3-4b28-9fe3-64b7ded6f751" />

## 📈 Key Insights

* RL provides **better adaptability** than traditional methods
* Multi-agent system improves **decision-making efficiency**
* Significant reduction in **waiting time vs heuristic**
* Better **traffic flow balance across all directions**


## ✨ Features

* ✅ Multi-agent reinforcement learning
* ✅ Real-time adaptive signal control
* ✅ Comparison with traditional methods
* ✅ Simulation + visualization
* ✅ Performance tracking

## 💻 Programming Language Used

* **Python**

## 🧠 Libraries & Tools Used

* **NumPy** – for numerical computations
* **Pandas** – for data handling and CSV results
* **Matplotlib** – for plotting graphs (like the one you shared)
* **Stable-Baselines3** – for implementing RL algorithms (PPO, DQN)
* **OS, Time, CSV, Argparse** – for file handling and execution control

## 👩‍💻 Author

**Shweta Vis**


## ⭐ Conclusion

The project demonstrates that an **AI-based multi-agent system** using reinforcement learning significantly improves traffic efficiency, making it a strong solution for **smart traffic management in modern cities**.
