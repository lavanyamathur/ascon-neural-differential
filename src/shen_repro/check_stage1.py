import numpy as np
import ascon as ac
from tensorflow.keras.models import load_model

np.random.seed(42)
model = load_model('saved_model/best4.h5')

X, Y = ac.make_td_diff(20000, 1, 4)
preds = model.predict(X, batch_size=5000, verbose=0).flatten()
acc = np.mean((preds > 0.5).astype(int) == Y)
print('Stage-1 round-4 val_acc on fresh data:', acc)