import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import ascon as ac

def main():
    group_size = 128
    num_rounds = 4
    train_groups = 78125
    eval_groups = 7812
    num_epochs = 50

    model = load_model('saved_model/best4.h5')

    model_s = Sequential()
    model_s.add(Dense(128, input_dim=group_size, activation='relu', kernel_regularizer=l2(10**-5)))
    model_s.add(Dense(128, activation='relu', kernel_regularizer=l2(10**-5)))
    model_s.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(10**-5)))
    model_s.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    scaler = MinMaxScaler(feature_range=(0, 1))

    print("Generating training groups...")
    X, Y = ac.make_td_diff(train_groups, group_size, num_rounds)
    X = np.sort(model.predict(X, batch_size=10000, verbose=0).reshape(train_groups, group_size))
    X = scaler.fit_transform(X.T).T

    print("Generating eval groups...")
    X_eval, Y_eval = ac.make_td_diff(eval_groups, group_size, num_rounds)
    X_eval = np.sort(model.predict(X_eval, batch_size=10000, verbose=0).reshape(eval_groups, group_size))
    X_eval = scaler.fit_transform(X_eval.T).T

    check = ModelCheckpoint('saved_model/best4_s=128_trained.h5', monitor='val_loss', save_best_only=True)
    h = model_s.fit(X, Y, epochs=num_epochs, batch_size=10000, shuffle=True,
                     validation_data=(X_eval, Y_eval), callbacks=[check])
    print("Best validation accuracy:", np.max(h.history['val_accuracy']))

if __name__ == '__main__':
    main()