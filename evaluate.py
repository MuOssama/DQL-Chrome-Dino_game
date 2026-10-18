"""Does the trained agent play forever, and did it find d*(speed, height)?
python evaluate.py --weights weights/weights.json --episodes 20 --max-steps 30000"""
import argparse
import numpy as np
from dino.env import DinoEnv
from dino.agents import AnalyticAgent, MLPAgent
from dino import physics as P

MAX_SCORE = 150   # speed ends at 5*1.1^15 = 21 px/frame

def run(agent, seed, max_steps):
    env = DinoEnv(seed, max_steps, max_score=MAX_SCORE); obs = env.reset(); rec = []
    while True:
        t = env.next_tree()
        a = agent.act(env, obs)
        if a == 1 and env.t_air is None:
            lo, hi = P.takeoff_window(env.s, t["h"], t["w"])
            rec.append((t["d"], lo, hi))     # d, safe-window low, d* (= window high)
        obs, _, done, info = env.step(a)
        if done: return env.frame, info["score"], info["dead"], rec

def report(agent, episodes, max_steps):
    fr, surv, recs = [], 0, []
    for s in range(episodes):
        f, sc, dead, rec = run(agent, s, max_steps)
        fr.append(f); surv += (not dead); recs += rec
    r = np.array(recs)
    d, lo, hi = r[:, 0], r[:, 1], r[:, 2]
    print(f"[{agent.name}] survived {surv}/{episodes} episodes, mean frames {np.mean(fr):.0f}")
    print(f"   jumps inside safe window: {np.mean((d >= lo) & (d <= hi)) * 100:.1f}%")
    print(f"   mean |d - d*| = {np.mean(np.abs(d - hi)):.1f}px   "
          f"extra landing distance vs minimum = {np.mean(hi - d):.1f}px")
    return surv == episodes, np.mean(np.abs(d - hi))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/weights.json")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=100000)
    a = p.parse_args()
    import os
    if not os.path.exists(a.weights):
        raise SystemExit(f"{a.weights} not found - run train.py first (it must print 'epoch ...' lines).")
    report(AnalyticAgent(), a.episodes, a.max_steps)
    ok, err = report(MLPAgent(a.weights), a.episodes, a.max_steps)
    print("\nVERDICT:", "RL plays infinitely and is close to the analytic d*." if ok and err < 12
          else "RL has NOT matched the analytic infinite-play function yet (train longer / tune).")
