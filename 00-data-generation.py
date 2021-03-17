import pickle
from LTI import LTI
from os import path
import numpy as np

np.random.seed(42)

import configs

DATA_DIR = configs.TRAIN_CONFIGS.get("data_dir")
LTI_FILE = configs.TRAIN_CONFIGS.get("lti_file")
STEPS = configs.TRAIN_CONFIGS.get("total_steps")


A = np.array([
    [1/3, 1/5, 2/7],
    [2/5, 2/9, 1/3],
    [1/2, 1/2, 1/8]
])
B = np.array([
    [1/2],
    [1/3],
    [3/5]
])

# initial hidden state
x0 = np.array([
    2.0,
    1.0,
    3/2
])

# # inputs
U = np.stack([np.random.uniform(0,2,size=B.shape[-1]) for _ in range(STEPS)])

# put in a container
lti = LTI(A, B, None, None, U, x0, STEPS)
# save the data
with open(path.join(DATA_DIR,LTI_FILE), "wb") as f:
    pickle.dump(lti, f)