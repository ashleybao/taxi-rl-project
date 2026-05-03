# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# class BiggerTaxiEnv(gym.Env):
#     def __init__(self, size=10):
#         super().__init__()
#         self.size = size

#         self.num_rows = size
#         self.num_cols = size

#         # 4 fixed locations (like Taxi)
#         self.locs = [(0,0), (0,size-1), (size-1,0), (size-1,size-1)]

#         self.action_space = spaces.Discrete(6)

#         # (row, col, passenger_loc, destination)
#         self.observation_space = spaces.Discrete(
#             self.num_rows * self.num_cols * 5 * 4
#         )

#         self.state = None

#     def encode(self, row, col, pass_loc, dest):
#         i = row
#         i *= self.num_cols
#         i += col
#         i *= 5
#         i += pass_loc
#         i *= 4
#         i += dest
#         return i

#     def decode(self, i):
#         out = []
#         out.append(i % 4)
#         i //= 4
#         out.append(i % 5)
#         i //= 5
#         out.append(i % self.num_cols)
#         i //= self.num_cols
#         out.append(i)
#         return reversed(out)

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)

#         row = np.random.randint(self.num_rows)
#         col = np.random.randint(self.num_cols)
#         pass_loc = np.random.randint(5)
#         dest = np.random.randint(4)

#         self.state = self.encode(row, col, pass_loc, dest)
#         mask = self.get_action_mask(row, col, pass_loc, dest)
#         return self.state, {"action_mask": mask}

#     def step(self, action):
#         row, col, pass_loc, dest = self.decode(self.state)

#         reward = -1
#         done = False

#         # movement
#         if action == 0 and row < self.num_rows - 1:
#             row += 1
#         elif action == 1 and row > 0:
#             row -= 1
#         elif action == 2 and col < self.num_cols - 1:
#             col += 1
#         elif action == 3 and col > 0:
#             col -= 1

#         # pickup
#         elif action == 4:
#             if pass_loc < 4 and (row, col) == self.locs[pass_loc]:
#                 pass_loc = 4
#             else:
#                 reward = -10

#         # dropoff
#         elif action == 5:
#             if pass_loc == 4 and (row, col) == self.locs[dest]:
#                 pass_loc = dest
#                 done = True
#                 reward = 20
#             else:
#                 reward = -10

#         self.state = self.encode(row, col, pass_loc, dest)
#         mask = self.get_action_mask(row, col, pass_loc, dest)
#         return self.state, reward, done, False, {"action_mask": mask}
    
#     def get_action_mask(self, row, col, pass_loc, dest):
#         mask = np.ones(6, dtype=np.int8)

#         # movement constraints
#         if row == 0:
#             mask[1] = 0  # north
#         if row == self.num_rows - 1:
#             mask[0] = 0  # south
#         if col == 0:
#             mask[3] = 0  # west
#         if col == self.num_cols - 1:
#             mask[2] = 0  # east

#         # pickup
#         if not (pass_loc < 4 and (row, col) == self.locs[pass_loc]):
#             mask[4] = 0

#         # dropoff
#         if not (pass_loc == 4 and (row, col) == self.locs[dest]):
#             mask[5] = 0

#         return mask

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BiggerTaxiEnv(gym.Env):
    def __init__(self, size=10):
        super().__init__()
        self.size = size
        self.num_rows = size
        self.num_cols = size

        # 4 fixed locations (like Taxi)
        self.locs = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]

        self.action_space = spaces.Discrete(6)

        # (row, col, passenger_loc, destination)
        self.observation_space = spaces.Discrete(
            self.num_rows * self.num_cols * 5 * 4
        )

        self.state = None

    def encode(self, row, col, pass_loc, dest):
        i = row
        i *= self.num_cols
        i += col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i //= 4
        out.append(i % 5)
        i //= 5
        out.append(i % self.num_cols)
        i //= self.num_cols
        out.append(i)
        return list(reversed(out))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        row = np.random.randint(self.num_rows)
        col = np.random.randint(self.num_cols)
        pass_loc = np.random.randint(4)
        dest = np.random.randint(4)

        self.state = self.encode(row, col, pass_loc, dest)
        mask = self.get_action_mask(row, col, pass_loc, dest)
        return self.state, {"action_mask": mask}

    def step(self, action):
        row, col, pass_loc, dest = self.decode(self.state)

        reward = -1
        done = False

        # movement
        if action == 0 and row < self.num_rows - 1:
            row += 1
        elif action == 1 and row > 0:
            row -= 1
        elif action == 2 and col < self.num_cols - 1:
            col += 1
        elif action == 3 and col > 0:
            col -= 1

        # pickup
        elif action == 4:
            if pass_loc < 4 and (row, col) == self.locs[pass_loc]:
                pass_loc = 4
            else:
                reward = -10

        # dropoff
        elif action == 5:
            if pass_loc == 4 and (row, col) == self.locs[dest]:
                pass_loc = dest
                done = True
                reward = 20
            else:
                reward = -10

        self.state = self.encode(row, col, pass_loc, dest)
        mask = self.get_action_mask(row, col, pass_loc, dest)
        return self.state, reward, done, False, {"action_mask": mask}

    def get_action_mask(self, row, col, pass_loc, dest):
        mask = np.ones(6, dtype=np.int8)

        if row == 0:
            mask[1] = 0  # north
        if row == self.num_rows - 1:
            mask[0] = 0  # south
        if col == 0:
            mask[3] = 0  # west
        if col == self.num_cols - 1:
            mask[2] = 0  # east

        if not (pass_loc < 4 and (row, col) == self.locs[pass_loc]):
            mask[4] = 0

        if not (pass_loc == 4 and (row, col) == self.locs[dest]):
            mask[5] = 0

        return mask

    def render(self):
        row, col, pass_loc, dest = self.decode(self.state)

        # RGB colors
        BLACK   = [0,   0,   0  ]
        # pastel tints for the 4 corner locations: R, G, B, Y
        LOC_TINTS = [
            [255, 180, 180],
            [180, 255, 180],
            [180, 180, 255],
            [255, 255, 150],
        ]
        TAXI    = [255, 220, 0  ]  # yellow — no passenger
        TAXI_ON = [0,   200, 0  ]  # green  — passenger aboard
        PASS    = [160, 0,   200]  # purple — passenger waiting
        DEST    = [255, 140, 0  ]  # orange — destination

        cell = 20  # pixels per cell
        frame = np.full((self.num_rows * cell, self.num_cols * cell, 3), 255, dtype=np.uint8)

        def fill(r, c, color):
            frame[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = color

        # 1. corner location tints
        for idx, (lr, lc) in enumerate(self.locs):
            fill(lr, lc, LOC_TINTS[idx])

        # 2. destination marker (overwrites tint)
        dr, dc = self.locs[dest]
        fill(dr, dc, DEST)

        # 3. passenger waiting (overwrites destination tint if same cell)
        if pass_loc < 4:
            pr, pc = self.locs[pass_loc]
            fill(pr, pc, PASS)

        # 4. taxi (always on top)
        fill(row, col, TAXI_ON if pass_loc == 4 else TAXI)

        # 5. grid lines
        for i in range(self.num_rows + 1):
            frame[i * cell: i * cell + 1, :] = BLACK
        for j in range(self.num_cols + 1):
            frame[:, j * cell: j * cell + 1] = BLACK

        return frame