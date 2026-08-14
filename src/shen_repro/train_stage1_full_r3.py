import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
import ascon as ac

def main():
    num_rounds = 3
    n = 10**7
    m = 10**6
    num_epochs = 50
    batch_size = 10000

    print("Generating training data (10 million samples)...")
    X, Y = ac.make_td_diff(n, 1, num_rounds)
    print("Generating eval data (1 million samples)...")
    X_eval, Y_eval = ac.make_td_diff(m, 1, num_rounds)

    model = Sequential()
    model.add(Dense(320, input_dim=320, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(320, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(320, activation='relu', kernel_regularizer=l2(10**-5)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Saves the best-val_accuracy epoch as training goes, so a crash or
    # interruption partway through doesn't lose the whole run.
    checkpoint = ModelCheckpoint(
        'ascon_stage1_full_10M_best_r3.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    h = model.fit(X, Y, epochs=num_epochs, batch_size=batch_size, shuffle=True,
                  validation_data=(X_eval, Y_eval), callbacks=[checkpoint])

    print("Fresh Stage-1 (FULL SCALE, 10M samples) best val_accuracy:", np.max(h.history['val_accuracy']))

    # Explicit save at the end too, in case save_best_only missed the final
    # epoch or you want the last-epoch weights specifically.
    model.save('ascon_stage1_full_10M_final_r3.keras')
    print("Final-epoch model saved to ascon_stage1_full_10M_final_r3.keras")
    print("Best-val_accuracy model saved to ascon_stage1_full_10M_best_r3.keras")

if __name__ == '__main__':
    main()
