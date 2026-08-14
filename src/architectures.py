"""
Four architectures for the rounds/bit-size sweep, matching the comparison
already planned for the ASCON project: MLP, Gohr-style shallow CNN,
Gohr-style deep ResNet, and DBitNet (dilated convolutions, no pooling).
Each builder takes input_dim (640 for full 2-pair ciphertext vectors,
or fewer if you're feeding a bit-subset from the bit-size sweep) and
returns a compiled Keras model.
"""
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Conv1D, Input, Reshape, Permute, Add, Flatten,
    BatchNormalization, Activation, Dropout
)
from tensorflow.keras.regularizers import l2


def build_mlp(input_dim=640, reg=1e-5):
    model = Sequential([
        Dense(320, activation='relu', input_dim=input_dim, kernel_regularizer=l2(reg)),
        Dense(320, activation='relu', kernel_regularizer=l2(reg)),
        Dense(320, activation='relu', kernel_regularizer=l2(reg)),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(reg)),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model


def build_gohr_cnn(input_dim=640, num_words=10, word_size=64,
                    num_filters=32, ks=3, reg=1e-5):
    """Shallow single-block version of Gohr's conv architecture."""
    inp = Input(shape=(input_dim,))
    rs = Reshape((num_words, word_size))(inp)
    perm = Permute((2, 1))(rs)
    conv = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg))(perm)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    flat = Flatten()(conv)
    d1 = Dense(64, kernel_regularizer=l2(reg))(flat)
    d1 = BatchNormalization()(d1)
    d1 = Activation('relu')(d1)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg))(d1)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model


def build_resnet(input_dim=640, num_words=10, word_size=64,
                  num_filters=32, num_outputs=1, d1=64, d2=64,
                  ks=3, depth=5, reg=1e-5):
    """Gohr's full published depth-residual architecture (Speck32/64 paper,
    adapted here for ASCON's word size/count)."""
    inp = Input(shape=(input_dim,))
    rs = Reshape((num_words, word_size))(inp)
    perm = Permute((2, 1))(rs)
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    for _ in range(depth):
        c1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg))(shortcut)
        c1 = BatchNormalization()(c1)
        c1 = Activation('relu')(c1)
        c2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg))(c1)
        c2 = BatchNormalization()(c2)
        c2 = Activation('relu')(c2)
        shortcut = Add()([shortcut, c2])
    flat = Flatten()(shortcut)
    dd1 = Dense(d1, kernel_regularizer=l2(reg))(flat)
    dd1 = BatchNormalization()(dd1)
    dd1 = Activation('relu')(dd1)
    dd2 = Dense(d2, kernel_regularizer=l2(reg))(dd1)
    dd2 = BatchNormalization()(dd2)
    dd2 = Activation('relu')(dd2)
    out = Dense(num_outputs, activation='sigmoid', kernel_regularizer=l2(reg))(dd2)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model


def build_dbitnet(input_dim=640, num_words=10, word_size=64,
                   num_filters=32, depth=4, reg=1e-5):
    """DBitNet-style architecture (Bellini et al.): dilated convolutions,
    no pooling, exponentially increasing dilation rate per block, so the
    receptive field grows without discarding positional resolution."""
    inp = Input(shape=(input_dim,))
    rs = Reshape((num_words, word_size))(inp)
    perm = Permute((2, 1))(rs)
    x = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg))(perm)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for i in range(depth):
        dilation = 2 ** i
        c = Conv1D(num_filters, kernel_size=3, padding='same',
                   dilation_rate=dilation, kernel_regularizer=l2(reg))(x)
        c = BatchNormalization()(c)
        c = Activation('relu')(c)
        x = Add()([x, c])
    flat = Flatten()(x)
    d = Dense(64, kernel_regularizer=l2(reg))(flat)
    d = BatchNormalization()(d)
    d = Activation('relu')(d)
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(reg))(d)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    return model


ARCHITECTURES = {
    'mlp': build_mlp,
    'gohr_cnn': build_gohr_cnn,
    'resnet': build_resnet,
    'dbitnet': build_dbitnet,
}
