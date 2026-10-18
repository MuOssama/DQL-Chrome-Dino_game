import json
import numpy as np
from . import physics as P

SAFETY = 2.0  # px

class HumanLikeNever: pass

class AnalyticAgent:
    """Jump when the gap reaches d*(speed, height) -> lands at minimal distance."""
    name = "analytic"
    def act(self, env, obs):
        t = env.next_tree()
        if t is None or env.t_air is not None:
            return 0
        return 1 if t["d"] <= P.optimal_takeoff(env.s, t["h"], t["w"]) - SAFETY else 0

class MLPAgent:
    """Numpy forward pass of the weights exported by train.py (same as the JS one)."""
    name = "rl"
    def __init__(self, path):
        self.layers = [(np.array(l["W"], dtype=np.float32), np.array(l["b"], dtype=np.float32))
                       for l in json.load(open(path))["layers"]]
    def q(self, obs):
        x = obs
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        return x
    def act(self, env, obs):
        if env.t_air is not None:
            return 0
        return int(np.argmax(self.q(obs)))

def export_json(torch_layers, path):
    layers = [{"W": l.weight.detach().cpu().numpy().tolist(),
               "b": l.bias.detach().cpu().numpy().tolist()} for l in torch_layers]
    json.dump({"layers": layers, "activation": "relu"}, open(path, "w"))
