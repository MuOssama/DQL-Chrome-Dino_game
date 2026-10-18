"""Shared game constants (the browser game in web/game.js uses the same numbers)."""
W = 600            # world width (px)
DINO_X = 50        # dino left edge
DINO_W = 30
DINO_H = 40
G = 0.6            # gravity (px / frame^2)
V0 = 12.0          # jump velocity (px / frame)  -> airtime 2*V0/G = 40 frames, apex 120 px
S0 = 5.0           # start speed (px / frame)
S_GROWTH = 1.1     # speed (fps) x1.1 ...
EVERY = 10         # ... every 10 obstacles passed
S_REF = 16.0       # observation normaliser
TREE_H = (20, 70)
TREE_W = (14, 30)
MARGIN = 30        # extra safety px in tree spacing
EXTRA = 150        # random extra spacing (px)
FPS = 60
OBS_DIM = 11
