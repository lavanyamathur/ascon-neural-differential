import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import ascon as ac

def main():
    num_rounds = 4
    n = 2000000
    m = 200000
    num_epochs = 30
    batch_size = 10000

    print("Generating training data...")
    X, Y = ac.make_td_diff(n, 1, num_rounds)
    print("Generating eval data...")
    X_eval, Y_eval = ac.make_td_diff(m, 1, num_rounds)

    model = Sequential()
    model.add(Dense(320, input_dim=320, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(320, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(320, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    h = model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                  validation_data=(X_eval, Y_eval))
    print("Fresh Stage-1 (2M samples) best val_accuracy:", np.max(h.history['val_accuracy']))

if __name__ == '__main__':
    main()