# benchmark_compare.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs all 3 controllers (Round-Robin, Heuristic, RL/DQN) through the headless
# simulation for N episodes each, then:
#   • Saves per-episode results to results/results.csv
#   • Saves summary stats  to results/summary.csv
#   • Plots a publication-quality comparison chart → results/comparison.png
#   • Prints a formatted final table to console
#
# Usage:
#   python benchmark_compare.py            # 5 episodes each (fast demo)
#   python benchmark_compare.py --episodes 20   # for your final submission
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from traffic_env import TrafficEnv

import headless_sim as sim

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=5,
                    help='Number of episodes per controller (default 5)')
parser.add_argument('--model', type=str, default=None,
                    help='Path to DQN model file (auto-detects if omitted)')
args = parser.parse_args()
N_EPISODES = args.episodes

os.makedirs('results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Controller definitions
# ─────────────────────────────────────────────────────────────────────────────
def roundrobin_action(tick, _queues):
    return tick % 4


def heuristic_action(_tick, queues):
    """Weighted score by vehicle type — same logic as original setTime()."""
    scores = []
    for i in range(4):
        d = sim.directionNumbers[i]
        c = bi = bu = tr = ri = 0
        for lane in range(3):
            for v in sim.vehicles[d][lane]:
                if v['crossed'] == 0:
                    t = v['type']
                    if   t == 'car':      c  += 1
                    elif t == 'bike':     bi += 1
                    elif t == 'bus':      bu += 1
                    elif t == 'truck':    tr += 1
                    elif t == 'rickshaw': ri += 1
        score = (c*sim.carTime + bi*sim.bikeTime + bu*sim.busTime +
                 tr*sim.truckTime + ri*sim.rickshawTime) / (sim.noOfLanes + 1)
        scores.append(score)
    return int(np.argmax(scores))


# Load RL model once
_rl_model = None
_rl_env = None


def _load_rl():
    global _rl_model, _rl_env

    if _rl_model is not None:
        return _rl_model

    model_path = './best_model/best_model.zip'   #  use BEST model
    norm_path  = 'vec_normalize.pkl'

    if os.path.exists(model_path) and os.path.exists(norm_path):
        from stable_baselines3 import DQN

        def make_env():
            return TrafficEnv()

        env = DummyVecEnv([make_env])
        env = VecNormalize.load(norm_path, env)

        env.training = False
        env.norm_reward = False

        _rl_model = DQN.load(model_path, env=env)
        _rl_env = env

        print("✅ RL model + normalization loaded correctly")
        return _rl_model

    print("❌ No trained model found")
    return None

def rl_action(_tick, queues):
    global _rl_env

    model = _load_rl()
    phase_time = getattr(rl_action, '_phase_time', 0)

   # 🔥 Build FULL observation (10 features)

    # 1. queue lengths
    q = queues

    # 2. average waiting time (same logic as env)
    avg_wait = []
    for i in range(4):
        d = sim.directionNumbers[i]
        waits = []
        for lane in range(3):
            for v in sim.vehicles[d][lane]:
                if not v['crossed']:
                    waits.append(v['wait_ticks'])
        avg_wait.append(np.mean(waits) if waits else 0)

    # 3. final observation
    obs = np.array(q + avg_wait + [sim.currentGreen, phase_time], dtype=np.float32)

    if model is None:
        return int(np.argmax(queues))

    # Convert to batch
    obs = obs.reshape((1, -1))

    # 🔥 CRITICAL: normalize observation
    obs = _rl_env.normalize_obs(obs)

    action, _ = model.predict(obs, deterministic=True)
    action = int(action[0])

    rl_action._phase_time = phase_time + 1 if action == sim.currentGreen else 0
    return action

rl_action._phase_time = 0

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Episode runner  (returns rich per-episode stats)
# ─────────────────────────────────────────────────────────────────────────────
def run_episode(action_fn):
    """
    Returns dict with:
      vehicles_crossed, throughput, total_wait, avg_wait_per_vehicle,
      max_queue, per_direction_crossed, per_direction_avg_wait
    """
    sim.reset_globals()
    tick = 0
    total_wait = 0
    max_queue  = 0

    done = False
    while not done:
        queues = sim.get_queue_lengths()
        action = action_fn(tick, queues)
        _, wait, done = sim.step_tick(action)
        total_wait += wait
        max_queue   = max(max_queue, max(queues))
        tick += 1

    # Collect final stats
    crossed_per_dir = [
        sim.vehicles[sim.directionNumbers[i]]['crossed'] for i in range(4)
    ]
    total_crossed = sum(crossed_per_dir)

    # Per-direction average wait
    avg_wait_per_dir = []
    for i in range(4):
        d = sim.directionNumbers[i]
        waits = [v['wait_ticks'] for lane in range(3)
                 for v in sim.vehicles[d][lane]]
        avg_wait_per_dir.append(np.mean(waits) if waits else 0.0)

    avg_wait_vehicle = total_wait / max(total_crossed, 1)

    return {
        'vehicles_crossed':       total_crossed,
        'throughput':             total_crossed / sim.simTime,
        'total_wait':             total_wait,
        'avg_wait_per_vehicle':   avg_wait_vehicle,
        'max_queue':              max_queue,
        'per_direction_crossed':  crossed_per_dir,
        'per_direction_avg_wait': avg_wait_per_dir,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Run all controllers
# ─────────────────────────────────────────────────────────────────────────────
controllers = [
    ('Round-Robin',  roundrobin_action),
    ('Heuristic',    heuristic_action),
    ('RL Agent',     rl_action),
]

all_results = {}   # name → list of episode dicts

print(f"\n{'━'*60}")
print(f"  TRAFFIC CONTROLLER BENCHMARK  ({N_EPISODES} episodes each)")
print(f"{'━'*60}\n")

for name, fn in controllers:
    print(f"Running: {name} ...")
    episodes = []
    rl_action._phase_time = 0  # reset between controllers
    for ep in range(N_EPISODES):
        t0 = time.time()
        result = run_episode(fn)
        elapsed = time.time() - t0
        episodes.append(result)
        print(f"  ep {ep+1:02d}  crossed={result['vehicles_crossed']:4d}  "
              f"throughput={result['throughput']:.3f}/s  "
              f"avg_wait={result['avg_wait_per_vehicle']:.1f}  "
              f"({elapsed:.1f}s)")
    all_results[name] = episodes
    print()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Save per-episode CSV
# ─────────────────────────────────────────────────────────────────────────────
csv_path = 'results/results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'controller', 'episode',
        'vehicles_crossed', 'throughput',
        'total_wait', 'avg_wait_per_vehicle', 'max_queue',
        'crossed_right', 'crossed_down', 'crossed_left', 'crossed_up',
        'avgwait_right', 'avgwait_down', 'avgwait_left', 'avgwait_up',
    ])
    for name, episodes in all_results.items():
        for ep_idx, r in enumerate(episodes):
            writer.writerow([
                name, ep_idx + 1,
                r['vehicles_crossed'], round(r['throughput'], 4),
                r['total_wait'], round(r['avg_wait_per_vehicle'], 2),
                r['max_queue'],
                *r['per_direction_crossed'],
                *[round(x, 2) for x in r['per_direction_avg_wait']],
            ])
print(f"Per-episode results saved → {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Compute summary stats
# ─────────────────────────────────────────────────────────────────────────────
summary = {}
for name, episodes in all_results.items():
    def _mean(key): return np.mean([e[key] for e in episodes])
    def _std(key):  return np.std( [e[key] for e in episodes])
    summary[name] = {
        'mean_crossed':    _mean('vehicles_crossed'),
        'std_crossed':     _std('vehicles_crossed'),
        'mean_throughput': _mean('throughput'),
        'std_throughput':  _std('throughput'),
        'mean_total_wait': _mean('total_wait'),
        'std_total_wait':  _std('total_wait'),
        'mean_avg_wait':   _mean('avg_wait_per_vehicle'),
        'std_avg_wait':    _std('avg_wait_per_vehicle'),
        'mean_max_queue':  _mean('max_queue'),
    }

summary_path = 'results/summary.csv'
with open(summary_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'controller',
        'mean_crossed','std_crossed',
        'mean_throughput','std_throughput',
        'mean_total_wait','std_total_wait',
        'mean_avg_wait_per_vehicle','std_avg_wait',
        'mean_max_queue',
    ])
    for name, s in summary.items():
        writer.writerow([name,
            round(s['mean_crossed'],1),    round(s['std_crossed'],1),
            round(s['mean_throughput'],4), round(s['std_throughput'],4),
            round(s['mean_total_wait'],0), round(s['std_total_wait'],0),
            round(s['mean_avg_wait'],2),   round(s['std_avg_wait'],2),
            round(s['mean_max_queue'],1),
        ])
print(f"Summary stats saved        → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plot
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'Round-Robin': '#ef4444',
    'Heuristic':   '#f59e0b',
    'RL Agent':    '#22c55e',
}
names   = list(summary.keys())
x       = np.arange(len(names))
width   = 0.55
bar_colors = [COLORS[n] for n in names]

fig = plt.figure(figsize=(18, 11), facecolor='#0f172a')
fig.suptitle('Traffic Signal Controller — Benchmark Comparison',
             color='white', fontsize=18, fontweight='bold', y=0.97)

gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
              left=0.06, right=0.97, top=0.90, bottom=0.08)

ax_style = dict(facecolor='#1e293b', grid_color='#334155')

def styled_ax(ax, title):
    ax.set_facecolor(ax_style['facecolor'])
    ax.tick_params(colors='#94a3b8', labelsize=9)
    ax.set_title(title, color='#e2e8f0', fontsize=11, pad=8, fontweight='semibold')
    ax.spines[:].set_color('#334155')
    ax.yaxis.label.set_color('#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.grid(axis='y', color='#334155', linewidth=0.6, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

# ── Chart 1: Vehicles Crossed ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
styled_ax(ax1, '① Vehicles Crossed (higher = better)')
vals = [summary[n]['mean_crossed'] for n in names]
errs = [summary[n]['std_crossed']  for n in names]
bars = ax1.bar(x, vals, width, color=bar_colors, yerr=errs,
               capsize=5, error_kw={'ecolor':'#64748b','lw':1.5})
ax1.set_xticks(x); ax1.set_xticklabels(names, color='#e2e8f0', fontsize=9)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val:.0f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

# ── Chart 2: Throughput ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
styled_ax(ax2, '② Throughput  vehicles/sec (higher = better)')
vals = [summary[n]['mean_throughput'] for n in names]
errs = [summary[n]['std_throughput']  for n in names]
bars = ax2.bar(x, vals, width, color=bar_colors, yerr=errs,
               capsize=5, error_kw={'ecolor':'#64748b','lw':1.5})
ax2.set_xticks(x); ax2.set_xticklabels(names, color='#e2e8f0', fontsize=9)
for bar, val in zip(bars, vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

# ── Chart 3: Avg Wait per Vehicle ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
styled_ax(ax3, '③ Avg Wait / Vehicle  ticks (lower = better)')
vals = [summary[n]['mean_avg_wait'] for n in names]
errs = [summary[n]['std_avg_wait']  for n in names]
bars = ax3.bar(x, vals, width, color=bar_colors, yerr=errs,
               capsize=5, error_kw={'ecolor':'#64748b','lw':1.5})
ax3.set_xticks(x); ax3.set_xticklabels(names, color='#e2e8f0', fontsize=9)
for bar, val in zip(bars, vals):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

# ── Chart 4: Total Wait ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
styled_ax(ax4, '④ Total Wait (lower = better)')
vals = [summary[n]['mean_total_wait'] for n in names]
errs = [summary[n]['std_total_wait']  for n in names]
bars = ax4.bar(x, vals, width, color=bar_colors, yerr=errs,
               capsize=5, error_kw={'ecolor':'#64748b','lw':1.5})
ax4.set_xticks(x); ax4.set_xticklabels(names, color='#e2e8f0', fontsize=9)
for bar, val in zip(bars, vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f'{val:.0f}', ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')

# ── Chart 5: Per-direction crossed (grouped bar, RL only for clean look) ──────
ax5 = fig.add_subplot(gs[1, 1])
styled_ax(ax5, '⑤ Vehicles Crossed per Direction')
dirs = ['Right', 'Down', 'Left', 'Up']
dir_x = np.arange(4)
bw = 0.25
for ci, (name, color) in enumerate(COLORS.items()):
    means = [np.mean([e['per_direction_crossed'][di] for e in all_results[name]])
             for di in range(4)]
    ax5.bar(dir_x + (ci-1)*bw, means, bw, label=name, color=color, alpha=0.85)
ax5.set_xticks(dir_x); ax5.set_xticklabels(dirs, color='#e2e8f0', fontsize=9)
ax5.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155',
           labelcolor='#e2e8f0', loc='upper right')

# ── Chart 6: Episode-over-episode line (throughput) ───────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
styled_ax(ax6, '⑥ Throughput Stability Across Episodes')
for name, color in COLORS.items():
    ys = [e['throughput'] for e in all_results[name]]
    ax6.plot(range(1, len(ys)+1), ys, 'o-', color=color,
             label=name, linewidth=2, markersize=5)
ax6.set_xlabel('Episode', color='#94a3b8', fontsize=9)
ax6.set_xticks(range(1, N_EPISODES+1))
ax6.tick_params(colors='#94a3b8')
ax6.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155',
           labelcolor='#e2e8f0')

chart_path = 'results/comparison.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Chart saved                → {chart_path}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Console summary table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*72}")
print(f"{'FINAL BENCHMARK RESULTS':^72}")
print(f"{'━'*72}")
print(f"{'Controller':<16} {'Crossed':>9} {'Throughput':>12} {'Avg Wait':>10} {'Total Wait':>12}")
print(f"{'─'*72}")
for name in names:
    s = summary[name]
    print(f"{name:<16} "
          f"{s['mean_crossed']:>7.1f}±{s['std_crossed']:<5.1f}"
          f"{s['mean_throughput']:>9.3f}/s   "
          f"{s['mean_avg_wait']:>7.1f}     "
          f"{s['mean_total_wait']:>9.0f}")
print(f"{'─'*72}")

# ── RL improvement %
best_base_crossed = max(summary['Round-Robin']['mean_crossed'],
                        summary['Heuristic']['mean_crossed'])
rl_crossed = summary['RL Agent']['mean_crossed']
improvement = (rl_crossed - best_base_crossed) / best_base_crossed * 100

best_base_wait = min(summary['Round-Robin']['mean_avg_wait'],
                     summary['Heuristic']['mean_avg_wait'])
rl_wait = summary['RL Agent']['mean_avg_wait']
wait_improvement = (best_base_wait - rl_wait) / best_base_wait * 100

print(f"\n  RL Agent improvement in throughput : {improvement:+.1f}% vs best baseline")
print(f"  RL Agent improvement in avg wait   : {wait_improvement:+.1f}% vs best baseline")
print(f"\n  All outputs saved in:  ./results/")
print(f"  ├── results.csv      (per-episode data)")
print(f"  ├── summary.csv      (mean ± std per controller)")
print(f"  └── comparison.png   (6-panel chart)")
print(f"{'━'*72}\n")
