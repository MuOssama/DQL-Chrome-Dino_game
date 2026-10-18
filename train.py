"""DQN training.  python train.py --epochs 1500   (1 epoch = 1 episode)"""
import faulthandler, os
faulthandler.enable()                       # prints a trace if Python itself crashes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse, random, shutil, time
from collections import deque
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(1)                    # avoids silent crashes/slowdowns with tiny nets on Windows
from dino.env import DinoEnv
from dino.config import OBS_DIM
from dino.agents import export_json

def evaluate(net, seed=1234, max_steps=3000):
    env = DinoEnv(seed, max_steps); obs = env.reset()
    while True:
        a = 0 if env.t_air is not None else int(net(torch.from_numpy(obs)).argmax())
        obs, _, done, info = env.step(a)
        if done: return env.frame, info["score"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--levels", type=int, default=10, help="start each episode at a random speed level 0..N (speed x1.1 per level)")
    p.add_argument("--eval-every", type=int, default=25)
    a = p.parse_args()
    print(f"torch {torch.__version__} | starting {a.epochs} epochs", flush=True)
    layers = [nn.Linear(OBS_DIM, 64), nn.Linear(64, 64), nn.Linear(64, 2)]
    net = nn.Sequential(layers[0], nn.ReLU(), layers[1], nn.ReLU(), layers[2])
    tgt = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    buf = deque(maxlen=100_000); rng = random.Random(0); steps = 0; best = (-1, -1)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("web", exist_ok=True)
    export_json(layers, "weights/weights.json")   # untrained placeholder so the file always exists
    for ep in range(1, a.epochs + 1):
        eps = max(0.02, 1 - ep / (0.6 * a.epochs))
        env = DinoEnv(max_steps=a.max_steps, start_level=rng.randint(0, a.levels)); obs = env.reset()
        while True:
            if env.t_air is not None: act = 0
            elif rng.random() < eps: act = 1 if rng.random() < 0.03 else 0
            else:
                with torch.no_grad(): act = int(net(torch.from_numpy(obs)).argmax())
            nobs, r, done, info = env.step(act)
            buf.append((obs, act, r, nobs, float(info["dead"])))
            obs = nobs; steps += 1
            if len(buf) > 2000 and steps % 4 == 0:
                b = rng.sample(buf, 128)
                o, ac, rw, no, d = (torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*b))
                q = net(o).gather(1, ac.long().unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    y = rw + 0.99 * tgt(no).max(1).values * (1 - d)
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10); opt.step()
            if steps % 500 == 0: tgt.load_state_dict(net.state_dict())
            if done: break
        if ep % a.eval_every == 0:
            fr, sc = evaluate(net)
            print(f"epoch {ep}/{a.epochs} eps={eps:.2f} train_score={info['score']} eval_frames={fr} eval_score={sc}", flush=True)
            if (fr, sc) >= best:
                best = (fr, sc)
                torch.save(net.state_dict(), "weights/dino_dqn.pt")
                export_json(layers, "weights/weights.json")
                shutil.copy("weights/weights.json", "web/weights.json")
    print("best eval (frames, score):", best, "-> weights/weights.json and web/weights.json")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); input("Error - press Enter to close")
