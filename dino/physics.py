"""The 'infinite play' equations.

Dino height after t frames in the air:  y(t) = V0*t - G*t^2/2   (lands at T = 2*V0/G)
A tree of height h is cleared while y(t) >= h, i.e. for t in [t1, t2]:
    t1,2 = (V0 -/+ sqrt(V0^2 - 2*G*h)) / G
Let d = gap between the tree's left edge and the dino's right edge at take-off,
s = tree speed (px/frame), w = tree width.  The tree overlaps the dino for
    t in [d/s, (d + w + DINO_W)/s]
so the take-off is SAFE when
    s*t1  <=  d  <=  s*t2 - w - DINO_W                      (take-off window)
The landing distance past the tree is  s*T - (d + w + DINO_W).
It is minimal for the LATEST safe jump:
    d* = s*t2 - w - DINO_W      ->  min landing distance = s*t1
If we ALWAYS wait until d* before jumping, every landing is at the minimum
distance s*t1 past the tree, and the next tree is still at least d* away when we
land (so we can wait again) as long as the edge-to-edge spacing satisfies
    G_edge >= s*t1_prev + d*(s, h_next, w_next) + DINO_W   (+ lag / safety margin)
Then the game can be played forever with zero deaths.
"""
import math
from .config import *

def airtime():
    return 2 * V0 / G

def apex():
    return V0 * V0 / (2 * G)

def clear_times(h):
    disc = V0 * V0 - 2 * G * h
    if disc <= 0:
        raise ValueError("tree too tall to jump")
    r = math.sqrt(disc)
    return (V0 - r) / G, (V0 + r) / G

def takeoff_window(s, h, w):
    t1, t2 = clear_times(h)
    return s * t1, s * t2 - w - DINO_W

def optimal_takeoff(s, h, w):
    """d*: the latest safe jump distance = minimal landing distance."""
    return takeoff_window(s, h, w)[1]

def min_landing_gap(s, h):
    return s * clear_times(h)[0]

def min_spacing(s, h_prev, h_next, w_next):
    """Edge-to-edge spacing that guarantees the 'land minimal, jump latest' loop
    never breaks (the +1 frame covers the 1-frame decision lag)."""
    return s * (clear_times(h_prev)[0] + 1) + DINO_W + optimal_takeoff(s, h_next, w_next) + MARGIN
