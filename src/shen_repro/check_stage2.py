import numpy as np
import ascon as ac
from tensorflow.keras.models import load_model

def main():
    np.random.seed(1)
    model1 = load_model('saved_model/best4.h5')
    model_s32 = load_model('saved_model/best4_s=32.h5')

    n = 20000
    group = 32
    X_eval, Y_eval = ac.make_td_diff(n, group, 4)
    scores = model1.predict(X_eval, batch_size=20000, verbose=0).reshape(n, group)
    scores_sorted = np.sort(scores, axis=1)
    mn = scores_sorted.min(axis=1, keepdims=True)
    mx = scores_sorted.max(axis=1, keepdims=True)
    scores_scaled = (scores_sorted - mn) / (mx - mn + 1e-12)

    preds = model_s32.predict(scores_scaled, batch_size=2000, verbose=0).flatten()
    acc = np.mean((preds > 0.5).astype(int) == Y_eval)
    print('n groups:', n, 'Stage-2 (s=32) accuracy:', acc)
    print('real groups mean score:', scores[Y_eval==1].mean(), 'random groups mean score:', scores[Y_eval==0].mean())

if __name__ == '__main__':
    main()