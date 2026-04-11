# run_simulation.py — runs the original pygame simulation driven by the trained PPO agent

import threading
import time
import numpy as np
import pygame
import sys
import os

# ── Load the trained model ────────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    model = PPO.load("./best_model/best_model")
    print("Loaded best model from training.")
except:
    try:
        from stable_baselines3 import PPO
        model = PPO.load("traffic_ppo")
        print("Loaded traffic_ppo model.")
    except:
        from stable_baselines3 import DQN
        model = DQN.load("traffic_dqn_v2")
        print("Loaded DQN model.")

# ── Import everything from your original simulation ───────────────────────────
from simulation import (
    TrafficSignal, Vehicle, signals, vehicles, directionNumbers,
    noOfSignals, defaultGreen, defaultYellow, defaultRed,
    defaultMinimum, defaultMaximum, simTime,
    signalCoods, signalTimerCoods, vehicleCountCoods,
    stopLines, defaultStop, stops, speeds,
    currentGreen, currentYellow, nextGreen,
    timeElapsed, simulation, gap
)
import simulation as sim_module

# ── RL observation builder (matches traffic_env.py) ───────────────────────────
def get_obs():
    queues = []
    for i in range(noOfSignals):
        d = directionNumbers[i]
        q = sum(1 for lane in range(3)
                for v in sim_module.vehicles[d][lane] if v.crossed == 0)
        queues.append(q)

    avg_waits = []
    for i in range(noOfSignals):
        d = directionNumbers[i]
        waiting = []
        for lane in range(3):
            for v in sim_module.vehicles[d][lane]:
                if v.crossed == 0:
                    waiting.append(getattr(v, 'wait_ticks', 0))
        avg_waits.append(float(np.mean(waiting)) if waiting else 0.0)

    phase_time = getattr(get_obs, 'phase_time', 0)
    return np.array(queues + avg_waits +
                    [sim_module.currentGreen, phase_time, sim_module.timeElapsed],
                    dtype=np.float32)

get_obs.phase_time = 0

# ── RL agent signal controller (replaces repeat()/setTime()) ──────────────────
def rl_signal_controller():
    while True:
        obs = get_obs()
        action, _ = model.predict(obs, deterministic=True)
        chosen = int(action)

        # Update phase time tracker
        if chosen == sim_module.currentGreen:
            get_obs.phase_time += 1
        else:
            get_obs.phase_time = 0

        # Apply the chosen green signal
        sim_module.currentYellow = 1
        time.sleep(sim_module.defaultYellow)   # show yellow briefly
        sim_module.currentYellow = 0
        sim_module.currentGreen = chosen
        sim_module.nextGreen = (chosen + 1) % noOfSignals

        # Hold green for a minimum time so vehicles can actually cross
        hold = max(10, int(sum(
            1 for lane in range(3)
            for v in sim_module.vehicles[directionNumbers[chosen]][lane]
            if v.crossed == 0
        ) * 1.5))
        hold = min(hold, defaultMaximum)
        time.sleep(hold)

# ── Vehicle + time threads (same as original) ─────────────────────────────────
def generateVehicles():
    import random
    vehicleTypes = {0:'car',1:'bus',2:'truck',3:'rickshaw',4:'bike'}
    while True:
        vtype = random.randint(0, 4)
        lane = 0 if vtype == 4 else random.randint(0,1) + 1
        will_turn = 0
        if lane == 2:
            will_turn = 1 if random.randint(0,4) <= 2 else 0
        temp = random.randint(0, 999)
        a = [400, 800, 900, 1000]
        dir_num = 0 if temp < a[0] else (1 if temp < a[1] else (2 if temp < a[2] else 3))
        Vehicle(lane, vehicleTypes[vtype], dir_num, directionNumbers[dir_num], will_turn)
        time.sleep(0.75)

def simulationTime():
    while True:
        sim_module.timeElapsed += 1
        time.sleep(1)
        if sim_module.timeElapsed == simTime:
            total = sum(sim_module.vehicles[directionNumbers[i]]['crossed']
                        for i in range(noOfSignals))
            print(f'\nSimulation ended after {simTime}s')
            print(f'Total vehicles crossed: {total}')
            print(f'Throughput: {total/simTime:.2f} vehicles/sec')
            os._exit(1)

# ── Start threads ─────────────────────────────────────────────────────────────
threading.Thread(target=rl_signal_controller, daemon=True).start()
threading.Thread(target=generateVehicles,     daemon=True).start()
threading.Thread(target=simulationTime,        daemon=True).start()

# ── Pygame render loop (copied from your Main class) ─────────────────────────
pygame.init()
black = (0,0,0); white = (255,255,255)
screen = pygame.display.set_mode((1400, 800))
pygame.display.set_caption("AI Traffic Control — RL Agent")
background  = pygame.image.load('images/mod_int.png')
redSignal   = pygame.image.load('images/signals/red.png')
yellowSignal= pygame.image.load('images/signals/yellow.png')
greenSignal = pygame.image.load('images/signals/green.png')
font = pygame.font.Font(None, 30)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.blit(background, (0,0))

    for i in range(noOfSignals):
        if i == sim_module.currentGreen:
            if sim_module.currentYellow == 1:
                screen.blit(yellowSignal, signalCoods[i])
                signals[i].signalText = "SLOW"
            else:
                screen.blit(greenSignal, signalCoods[i])
                signals[i].signalText = "GO"
        else:
            screen.blit(redSignal, signalCoods[i])
            signals[i].signalText = "---"

    for i in range(noOfSignals):
        txt = font.render(str(signals[i].signalText), True, white, black)
        screen.blit(txt, signalTimerCoods[i])
        crossed = sim_module.vehicles[directionNumbers[i]]['crossed']
        ctxt = font.render(str(crossed), True, black, white)
        screen.blit(ctxt, vehicleCountCoods[i])

        # Show which controller is active
        agent_txt = font.render("RL AGENT CONTROLLING", True, (0,200,0), black)
        screen.blit(agent_txt, (500, 10))

        time_txt = font.render(f"Time: {sim_module.timeElapsed}s", True, black, white)
        screen.blit(time_txt, (1100, 50))

        for vehicle in simulation:
            screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
            vehicle.move()

        pygame.display.update()