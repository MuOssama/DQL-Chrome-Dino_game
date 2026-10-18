"""DQN in pure numpy (no PyTorch).  python train_numpy.py --epochs 1500"""
import argparse, json, os, shutil
import numpy as np
from dino.env import DinoEnv
from dino.config import OBS_DIM

class MLP:
    def __init__(s, sizes, rng):
        s.W = [(rng.standard_normal((o, i)) * np.sqrt(2 / i)).astype(np.float32) for i, o in zip(sizes[:-1], sizes[1:])]
        s.b = [np.zeros(o, np.float32) for o in sizes[1:]]
        s.m = [np.zeros_like(p) for p in s.W + s.b]; s.v = [np.zeros_like(p) for p in s.W + s.b]; s.t = 0
    def forward(s, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(s.W, s.b)):
            x = x @ W.T + b
            if i < len(s.W) - 1: x = np.maximum(x, 0)
            acts.append(x)
        return acts
    def q(s, x): return s.forward(x[None])[-1][0]
    def train(s, x, a, y, lr=1e-3):
        acts = s.forward(x); q = acts[-1]; B = len(x); idx = np.arange(B)
        g = np.zeros_like(q); g[idx, a] = np.clip(q[idx, a] - y, -1, 1) / B   # Huber gradient
        gW, gb = [None] * len(s.W), [None] * len(s.W)
        for i in reversed(range(len(s.W))):
            gW[i] = g.T @ acts[i]; gb[i] = g.sum(0)
            if i > 0: g = (g @ s.W[i]) * (acts[i] > 0)
        s.t += 1
        for k, (p, gr) in enumerate(zip(s.W + s.b, gW + gb)):   # Adam
            s.m[k] = 0.9 * s.m[k] + 0.1 * gr; s.v[k] = 0.999 * s.v[k] + 0.001 * gr * gr
            p -= lr * (s.m[k] / (1 - 0.9 ** s.t)) / (np.sqrt(s.v[k] / (1 - 0.999 ** s.t)) + 1e-8)
    def copy_from(s, o):
        s.W = [w.copy() for w in o.W]; s.b = [b.copy() for b in o.b]
    def export(s, path):
        json.dump({"layers": [{"W": w.tolist(), "b": b.tolist()} for w, b in zip(s.W, s.b)], "activation": "relu"}, open(path, "w"))

def evaluate(net, seeds=range(1000, 1005), max_steps=3000):
    sc, fr = [], []
    for sd in seeds:
        env = DinoEnv(sd, max_steps); obs = env.reset()
        while True:
            a = 0 if env.t_air is not None else int(np.argmax(net.q(obs)))
            obs, _, done, info = env.step(a)
            if done: sc.append(info["score"]); fr.append(env.frame); break
    return float(np.mean(sc)), float(np.mean(fr))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--levels", type=int, default=10, help="start each episode at a random speed level 0..N (speed x1.1 per level)")
    p.add_argument("--eval-every", type=int, default=25)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    net = MLP([OBS_DIM, 64, 64, 2], rng); tgt = MLP([OBS_DIM, 64, 64, 2], rng); tgt.copy_from(net)
    N = 100_000; O = np.zeros((N, OBS_DIM), np.float32); NO = O.copy()
    A = np.zeros(N, np.int64); R = np.zeros(N, np.float32); D = np.zeros(N, np.float32)
    GAM = np.zeros(N, np.float32); n = 0; best = (-1, -1); NSTEP, GAMMA = 30, 0.99
    wts = GAMMA ** np.arange(NSTEP)
    os.makedirs("weights", exist_ok=True); os.makedirs("web", exist_ok=True)
    net.export("weights/weights.json")
    print("pure-numpy n-step DQN, starting", a.epochs, "epochs", flush=True)
    decay = min(0.3 * a.epochs, 1500)
    for ep in range(1, a.epochs + 1):
        eps = max(0.02, 1 - ep / decay)
        env = DinoEnv(max_steps=a.max_steps, start_level=int(rng.integers(0, a.levels + 1))); obs = env.reset()
        S, Ac, Rw = [], [], []
        while True:
            if env.t_air is not None: act = 0
            elif rng.random() < eps: act = 1 if rng.random() < 0.03 else 0
            else: act = int(np.argmax(net.q(obs)))
            nobs, r, done, info = env.step(act)
            S.append(obs); Ac.append(act); Rw.append(r); obs = nobs
            if done: break
        L = len(S); S = np.array(S, np.float32); Rw = np.array(Rw, np.float32)
        G = np.correlate(np.concatenate([Rw, np.zeros(NSTEP - 1, np.float32)]), wts, "valid")
        allS = np.concatenate([S, obs[None]])            # index L = final state
        t = np.arange(L); boot = np.minimum(t + NSTEP, L)
        gm = np.where(boot == L, 0.0 if info["dead"] else 1.0, 1.0) * GAMMA ** (boot - t)
        gm = np.where((boot == L) & info["dead"], 0.0, GAMMA ** (boot - t)).astype(np.float32)
        idx = (n + t) % N
        O[idx], A[idx], R[idx], NO[idx], GAM[idx] = S, Ac, G, allS[boot], gm; n += L
        if n > 2000:
            for _ in range(max(1, L // 4)):
                b = rng.integers(0, min(n, N), 128)
                y = R[b] + GAM[b] * tgt.forward(NO[b])[-1].max(1)
                net.train(O[b], A[b], y)
            if ep % 5 == 0: tgt.copy_from(net)
        if ep % a.eval_every == 0:
            sc, fr = evaluate(net)
            print(f"epoch {ep}/{a.epochs} eps={eps:.2f} train_score={info['score']} eval_mean_score={sc:.1f} eval_mean_frames={fr:.0f}", flush=True)
            if (sc, fr) >= best:
                best = (sc, fr); net.export("weights/weights.json"); shutil.copy("weights/weights.json", "web/weights.json")
    print("best eval (mean score, mean frames):", best, "-> weights/weights.json and web/weights.json")

if __name__ == "__main__":
    main()
