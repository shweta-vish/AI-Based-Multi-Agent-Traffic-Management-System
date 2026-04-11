# compare_final_pygame.py
# Runs all 3 controllers through the real pygame simulation and compares results

import pygame
import sys
import os
import time
import random
import threading
import numpy as np

# ── Shared config (same as simulation.py) ────────────────────────────────────
defaultRed = 150; defaultYellow = 2; defaultGreen = 20
defaultMinimum = 10; defaultMaximum = 60
noOfSignals = 4; simTime = 300
speeds = {'car':2.25,'bus':1.8,'truck':1.8,'rickshaw':2,'bike':2.5}
carTime=2; bikeTime=1; rickshawTime=2.25; busTime=2.5; truckTime=2.5
noOfLanes = 2
directionNumbers = {0:'right',1:'down',2:'left',3:'up'}
vehicleTypes     = {0:'car',1:'bus',2:'truck',3:'rickshaw',4:'bike'}
signalCoods      = [(530,230),(810,230),(810,570),(530,570)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods= [(480,210),(880,210),(880,550),(480,550)]
stopLines  = {'right':590,'down':330,'left':800,'up':535}
defaultStop= {'right':580,'down':320,'left':810,'up':545}
gap = 15; gap2 = 15; rotationAngle = 3
mid = {'right':{'x':705,'y':445},'down':{'x':695,'y':450},
       'left':{'x':695,'y':425},'up':{'x':695,'y':400}}

def run_one_episode(controller_name, get_action_fn):
    """Run a full 300s pygame simulation with the given controller."""

    # ── per-episode state ─────────────────────────────────────────────────────
    from simulation import TrafficSignal
    signals = []
    signals.append(TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum))
    signals.append(TrafficSignal(signals[0].yellow+signals[0].green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum))
    signals.append(TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum))

    state = {
        'currentGreen': 0,
        'currentYellow': 0,
        'nextGreen': 1,
        'timeElapsed': 0,
        'done': False,
        'crossed': [0,0,0,0],
        'phase_time': 0,
    }

    x_pos  = {'right':[0,0,0],'down':[755,727,697],'left':[1400,1400,1400],'up':[602,627,657]}
    y_pos  = {'right':[348,370,398],'down':[0,0,0],'left':[498,466,436],'up':[800,800,800]}
    stops  = {'right':[580,580,580],'down':[320,320,320],'left':[810,810,810],'up':[545,545,545]}
    vehicles = {d:{0:[],1:[],2:[],'crossed':0} for d in directionNumbers.values()}
    sim_group = pygame.sprite.Group()

    # ── Vehicle class (local, clean) ──────────────────────────────────────────
    class Vehicle(pygame.sprite.Sprite):
        def __init__(self, lane, vclass, dir_num, direction, will_turn):
            pygame.sprite.Sprite.__init__(self)
            self.lane=lane; self.vehicleClass=vclass
            self.speed=speeds[vclass]; self.direction=direction
            self.dir_num=dir_num; self.will_turn=will_turn
            self.x=x_pos[direction][lane]; self.y=y_pos[direction][lane]
            self.crossed=0; self.turned=0; self.rotateAngle=0
            self.wait_ticks=0
            vehicles[direction][lane].append(self)
            self.index=len(vehicles[direction][lane])-1
            path=f"images/{direction}/{vclass}.png"
            self.originalImage=pygame.image.load(path)
            self.currentImage=pygame.image.load(path)
            idx=self.index
            prev=vehicles[direction][lane][idx-1] if idx>0 else None
            if direction=='right':
                self.stop = (prev.stop - prev.currentImage.get_rect().width - gap
                             if prev and prev.crossed==0 else defaultStop[direction])
                tmp=self.currentImage.get_rect().width+gap
                x_pos[direction][lane]-=tmp; stops[direction][lane]-=tmp
            elif direction=='left':
                self.stop = (prev.stop + prev.currentImage.get_rect().width + gap
                             if prev and prev.crossed==0 else defaultStop[direction])
                tmp=self.currentImage.get_rect().width+gap
                x_pos[direction][lane]+=tmp; stops[direction][lane]+=tmp
            elif direction=='down':
                self.stop = (prev.stop - prev.currentImage.get_rect().height - gap
                             if prev and prev.crossed==0 else defaultStop[direction])
                tmp=self.currentImage.get_rect().height+gap
                y_pos[direction][lane]-=tmp; stops[direction][lane]-=tmp
            elif direction=='up':
                self.stop = (prev.stop + prev.currentImage.get_rect().height + gap
                             if prev and prev.crossed==0 else defaultStop[direction])
                tmp=self.currentImage.get_rect().height+gap
                y_pos[direction][lane]+=tmp; stops[direction][lane]+=tmp
            sim_group.add(self)

        def move(self):
            cg = state['currentGreen']; cy = state['currentYellow']
            dn = self.dir_num; d = self.direction; lane = self.lane
            idx = self.index
            prev = vehicles[d][lane][idx-1] if idx>0 else None
            w = self.currentImage.get_rect().width
            h = self.currentImage.get_rect().height

            if d=='right':
                if self.crossed==0 and self.x+w>stopLines[d]:
                    self.crossed=1; vehicles[d]['crossed']+=1; state['crossed'][dn]+=1
                can_move=((self.x+w<=self.stop or self.crossed==1 or (cg==0 and cy==0)) and
                          (prev is None or self.x+w<prev.x-gap2 or prev.turned==1))
                if can_move: self.x+=self.speed
                else: self.wait_ticks+=1

            elif d=='left':
                if self.crossed==0 and self.x<stopLines[d]:
                    self.crossed=1; vehicles[d]['crossed']+=1; state['crossed'][dn]+=1
                can_move=((self.x>=self.stop or self.crossed==1 or (cg==2 and cy==0)) and
                          (prev is None or self.x>prev.x+prev.currentImage.get_rect().width+gap2 or prev.turned==1))
                if can_move: self.x-=self.speed
                else: self.wait_ticks+=1

            elif d=='down':
                if self.crossed==0 and self.y+h>stopLines[d]:
                    self.crossed=1; vehicles[d]['crossed']+=1; state['crossed'][dn]+=1
                can_move=((self.y+h<=self.stop or self.crossed==1 or (cg==1 and cy==0)) and
                          (prev is None or self.y+h<prev.y-gap2 or prev.turned==1))
                if can_move: self.y+=self.speed
                else: self.wait_ticks+=1

            elif d=='up':
                if self.crossed==0 and self.y<stopLines[d]:
                    self.crossed=1; vehicles[d]['crossed']+=1; state['crossed'][dn]+=1
                can_move=((self.y>=self.stop or self.crossed==1 or (cg==3 and cy==0)) and
                          (prev is None or self.y>prev.y+prev.currentImage.get_rect().height+gap2 or prev.turned==1))
                if can_move: self.y-=self.speed
                else: self.wait_ticks+=1

    # ── Signal control thread ─────────────────────────────────────────────────
    def signal_thread():
        while not state['done']:
            # Build observation
            queues = [sum(1 for lane in range(3)
                          for v in vehicles[directionNumbers[i]][lane] if v.crossed==0)
                      for i in range(4)]
            avg_waits = []
            for i in range(4):
                d = directionNumbers[i]
                wt = [v.wait_ticks for lane in range(3)
                      for v in vehicles[d][lane] if v.crossed==0]
                avg_waits.append(float(np.mean(wt)) if wt else 0.0)
            obs = np.array(queues + avg_waits +
                           [state['currentGreen'], state['phase_time'], state['timeElapsed']],
                           dtype=np.float32)

            chosen = get_action_fn(obs, state['timeElapsed'])

            if chosen == state['currentGreen']:
                state['phase_time'] += 1
            else:
                state['phase_time'] = 0

            # Yellow phase
            state['currentYellow'] = 1
            time.sleep(defaultYellow)
            state['currentYellow'] = 0
            state['currentGreen']  = chosen

            # Hold green proportional to queue
            hold = max(defaultMinimum,
                       min(int(queues[chosen] * 2), defaultMaximum))
            time.sleep(hold)

    # ── Vehicle spawn thread ──────────────────────────────────────────────────
    def spawn_thread():
        while not state['done']:
            vtype = random.randint(0,4)
            lane  = 0 if vtype==4 else random.randint(0,1)+1
            will_turn = 1 if (lane==2 and random.randint(0,4)<=2) else 0
            tmp = random.randint(0,999)
            dn  = 0 if tmp<400 else (1 if tmp<800 else (2 if tmp<900 else 3))
            Vehicle(lane, vehicleTypes[vtype], dn, directionNumbers[dn], will_turn)
            time.sleep(0.75)

    # ── Timer thread ──────────────────────────────────────────────────────────
    def timer_thread():
        while not state['done']:
            time.sleep(1)
            state['timeElapsed'] += 1
            if state['timeElapsed'] >= simTime:
                state['done'] = True

    threading.Thread(target=signal_thread, daemon=True).start()
    threading.Thread(target=spawn_thread,  daemon=True).start()
    threading.Thread(target=timer_thread,  daemon=True).start()

    # ── Pygame render loop ────────────────────────────────────────────────────
    black=(0,0,0); white=(255,255,255)
    background   = pygame.image.load('images/mod_int.png')
    redSignal    = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal  = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)
    clock = pygame.time.Clock()

    while not state['done']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        screen.blit(background,(0,0))

        for i in range(noOfSignals):
            if i==state['currentGreen']:
                img = yellowSignal if state['currentYellow'] else greenSignal
            else:
                img = redSignal
            screen.blit(img, signalCoods[i])
            txt = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(txt, signalTimerCoods[i])
            ctxt = font.render(str(vehicles[directionNumbers[i]]['crossed']),
                               True, black, white)
            screen.blit(ctxt, vehicleCountCoods[i])

        label = font.render(f"{controller_name}  |  Time: {state['timeElapsed']}s",
                            True, (0,220,0), black)
        screen.blit(label, (400, 10))

        for v in sim_group:
            screen.blit(v.currentImage, [v.x, v.y])
            v.move()

        pygame.display.update()
        clock.tick(60)

    total = sum(vehicles[directionNumbers[i]]['crossed'] for i in range(4))
    lane_counts = [vehicles[directionNumbers[i]]['crossed'] for i in range(4)]
    return total, lane_counts

# ── Controller action functions ───────────────────────────────────────────────
_rr_tick = [0]
def roundrobin_action(obs, tick):
    _rr_tick[0] += 1
    return _rr_tick[0] % 4

def heuristic_action(obs, tick):
    import headless_sim as sim
    queues = obs[:4]
    weights = [carTime*2 + bikeTime + busTime*2.5 + truckTime*2.5]
    return int(np.argmax(queues))   # pick busiest lane

_rl_model = [None]
def rl_action(obs, tick):
    if _rl_model[0] is None:
        from stable_baselines3 import PPO
        try:    _rl_model[0] = PPO.load("./best_model/best_model")
        except: _rl_model[0] = PPO.load("traffic_ppo")
    action, _ = _rl_model[0].predict(obs, deterministic=True)
    return int(np.array(action).flatten()[0])

# ── Main ──────────────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((1400, 800))
pygame.display.set_caption("Traffic Controller Comparison")

results = {}
controllers = [
    ("Fixed round-robin", roundrobin_action),
    ("Heuristic (setTime)", heuristic_action),
    ("RL agent (PPO)", rl_action),
]

for name, fn in controllers:
    _rr_tick[0] = 0
    print(f"\nRunning: {name} ...")
    pygame.display.set_caption(f"Running: {name}")
    total, lanes = run_one_episode(name, fn)
    results[name] = (total, lanes)
    print(f"  Crossed: {total}  |  Lanes: {lanes}")
    time.sleep(2)   # brief pause between episodes

# ── Final results ─────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"{'FINAL COMPARISON':^55}")
print(f"{'═'*55}")
print(f"{'Controller':<25} {'Crossed':>8} {'Throughput':>12}")
print(f"{'─'*55}")
for name, (total, lanes) in results.items():
    print(f"{name:<25} {total:>8}  {total/simTime:>10.3f}/s")
print(f"{'─'*55}")

best_base = max(results['Fixed round-robin'][0], results['Heuristic (setTime)'][0])
rl_total  = results['RL agent (PPO)'][0]
print(f"\nRL improvement over best baseline: {(rl_total-best_base)/best_base*100:+.1f}%")
pygame.quit()