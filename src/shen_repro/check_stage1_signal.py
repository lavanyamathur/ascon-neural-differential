import ascon as ac
from tensorflow.keras.models import load_model
import numpy as np

stage1 = load_model("./saved_model/ascon_stage1_full_10M_best.keras")
X, Y = ac.make_td_diff(2000, 128, 4)   # small sample, fast
scores = stage1.predict(X, batch_size=2000, verbose=0).reshape(2000, 128)

real_scores = scores[Y == 1].mean()
rand_scores = scores[Y == 0].mean()
print("mean score | real groups:", real_scores, " random groups:", rand_scores)
