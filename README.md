# DQL-Chrome-Dino_game
Chrome-style offline Dino game in Python. A closed-form “never lose” jump equation vs. a Deep Q-Learning agent (pure NumPy) that learns to play, plus a browser version that loads the trained weights.  Topics:  python reinforcement-learning deep-q-learning dqn numpy game-ai dino-game neural-network javascript
# From Math to Deep Q-Learning: Teaching a Dino to Never Lose
 
I rebuilt Chrome's "no internet" dino game in Python: only trees and a jumping dino, with the speed rising 10% every 10 obstacles. The same game runs in the browser and in training. I train a neural network on my laptop and load its weights into the game, so anyone can watch the AI play or play it themselves (space or ↑).
 
## 1. The math gives us a rule for never losing
 
The jump is physics: `y(t) = V0·t − G·t²/2`. A tree of height `h` is cleared between two moments:
 
```
t1,2 = (V0 ∓ √(V0² − 2·G·h)) / G
```
 
With tree speed `s` and width `w`, a jump started when the tree is `d` away is safe when:
 
```
s·t1  ≤  d  ≤  s·t2 − w − dino_width
```
 
The best jump is the latest safe one, `d* = s·t2 − w − dino_width`. It lands the dino at the minimum distance past the tree, which leaves the most room for the next tree. If trees are spaced far enough apart, always jumping at `d*` never loses. This is the **non-lose equation**, and it depends only on speed and tree height.
 
## 2. The real world breaks the rule
 
This equation exists because the game is simple: one jump, rectangular trees, one action. Add a bird at a different height, trees of changing width, or a head-down action, and the rules for never losing become very hard to write by hand. The mathematical model is too rigid to follow every new case.
 
## 3. So we let the machine learn the rule
 
- **Q-Learning (QL):** the agent keeps a table `Q(state, action)` of how good each action is, and improves it from rewards and deaths.
- **Deep Learning (DL):** a neural network learns patterns from data instead of following written rules.
- **Deep Q-Learning (DQL):** a neural network replaces the Q-table. This is needed because distances and speeds are continuous, and no table can hold every value.
The agent is never told the equation. It plays, loses, gets rewarded, and the network ends up holding the rule inside its weights.
 
## 4. How we check it learned the right thing
 
The equation is the judge. I run the equation player and the trained agent on the same games and compare:
 
- Does it survive long games (150 obstacles)?
- How often do its jumps fall in the safe window?
- How close is its jump distance to `d*`?
If the agent survives and sits near `d*`, it has rediscovered the equation on its own.
 
## Takeaway
 
Where the math model exists, it gives the exact answer and lets us verify the learner. Where the world is too complex to model, deep Q-learning learns the rule for us. The simple dino game is the place to prove that.
 
