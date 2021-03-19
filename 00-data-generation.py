import sys

import pickle
from LTI import LTI
from os import path
import numpy as np

np.random.seed(42)

import configs

DATA_DIR = configs.TRAIN_CONFIGS.get("data_dir")
LTI_FILE = configs.TRAIN_CONFIGS.get("lti_file")
STEPS = configs.TRAIN_CONFIGS.get("total_steps")

# TODO implement better way to organize experiments... 

# # 00- (target=hidden -- base=all )
# A = np.array(
#     [[1/3, 1/5, 2/7],
#      [2/5, 2/9, 1/3],
#      [1/2, 1/2, 1/8]]
# )
# B = np.array(
#     [[1/2],
#      [1/3],
#      [3/5]]
# )
# C = None
# D = None

# # 01- (target=hidden -- base=all )
# A = np.array([
#     [-2/9, 3/8, 5/7],
#     [1/7, -2/5, 3/7],
#     [1/3, 1/8, 5/8]
# ])
# B = np.array([
#     [1/2],
#     [2/3],
#     [4/5]
# ])
# C = None
# D = None


## 02- (target=output -- base=all)
## 03- (target=output -- base=all, layers=2, init_h=False)
A = np.array(
    [[1/3, 1/5, 2/7],
     [2/5, 2/9, 1/3],
     [1/2, 1/2, 1/8]]
)
B = np.array(
    [[1/2],
     [1/3],
     [3/5]]
)
C = np.array(
    [[2,1,3],
     [3,1,3],
     [2,2,5]]
)
D = None


# initial hidden state
x0 = np.array([
    2.0,
    1.0,
    3/2
])

# inputs
U = np.stack([np.random.uniform(0,2,size=B.shape[-1]) for _ in range(STEPS)])

eigen_values = np.linalg.eigvals(A)
if not (np.abs(eigen_values) < 1).all():
    resp = input(f"eigen_values of A='{eigen_values}'. System is unstable. Continue? (y/n) : ")
    if resp.lower() == "y":
        pass
    else:
        sys.exit()
else:
    print(f"eigen_values of A='{eigen_values}'")

# put in a container
lti = LTI(A, B, C, D, U, x0, STEPS)
# save the data
with open(path.join(DATA_DIR,LTI_FILE), "wb") as f:
    pickle.dump(lti, f)