import random
import numpy as np
from .config import *
from . import physics as P

class DinoEnv:
    """Headless game. Actions: 0 = nothing, 1 = jump (ignored in the air)."""
    def __init__(self, seed=None, max_steps=None, max_score=None, start_level=0):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.max_score = max_score
        self.start_level = start_level
        self.reset()

    def reset(self):
        self.t_air = None
        self.y = 0.0
        self.s = S0 * S_GROWTH ** self.start_level
        self.frame = 0
        self.score = 0
        self.trees = []          # [x, w, h, passed]
        self._fill_trees()
        return self.obs()

    def _spawn(self):
        h = self.rng.randint(*TREE_H); w = self.rng.randint(*TREE_W)
        if not self.trees:
            x = W + 100
        else:
            x0, w0, h0, _ = self.trees[-1]
            sf = self.s * S_GROWTH
            gap = P.min_spacing(sf, h0, h, w) + self.rng.uniform(0, EXTRA)
            x = x0 + w0 + gap
        self.trees.append([x, w, h, False])

    def _fill_trees(self):
        while len(self.ahead()) < 4:
            self._spawn()

    def ahead(self):
        return [t for t in self.trees if t[0] + t[1] > DINO_X]

    def next_tree(self):
        a = self.ahead()
        if not a:
            return None
        x, w, h, _ = a[0]
        return {"d": x - (DINO_X + DINO_W), "w": w, "h": h}

    def obs(self):
        a = self.ahead()[:2]
        f = []
        for x, w, h, _ in a:
            d = x - DINO_X - DINO_W
            f += [d / W, d / (self.s * P.airtime()), h / TREE_H[1], w / TREE_W[1]]
        f += [0.0] * (8 - len(f))
        air = 0.0 if self.t_air is None else self.t_air / P.airtime()
        f += [self.s / S_REF, self.y / P.apex(), air]
        return np.array(f, dtype=np.float32)

    def step(self, action):
        r = 0.01
        if action == 1 and self.t_air is None:
            self.t_air = 0
            r -= 0.05
        if self.t_air is not None:
            self.t_air += 1
            t = self.t_air
            y = V0 * t - 0.5 * G * t * t
            if y <= 0:
                y, self.t_air = 0.0, None
            self.y = y
        for t in self.trees:
            t[0] -= self.s
        self.frame += 1
        dead = any(x < DINO_X + DINO_W and x + w > DINO_X and self.y < h
                   for x, w, h, _ in self.trees)
        for t in self.trees:
            if not t[3] and t[0] + t[1] < DINO_X:
                t[3] = True; self.score += 1; r += 1.0
                if self.score % EVERY == 0:
                    self.s *= S_GROWTH
        self.trees = [t for t in self.trees if t[0] + t[1] > -50]
        self._fill_trees()
        trunc = (self.max_steps is not None and self.frame >= self.max_steps) or \
                (self.max_score is not None and self.score >= self.max_score)
        if dead:
            r = -5.0
        return self.obs(), r, dead or trunc, {"score": self.score, "dead": dead, "truncated": trunc}
