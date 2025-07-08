from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from custom_layers import mobilevit_block

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=15):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    bn = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)

    # Transformer
    mv = mobilevit_block(bn, patch_size=2, transformer_dim=64, num_heads=4, ff_dim=128)

    # Decoder
    u1 = UpSampling2D((2, 2))(mv)
    concat1 = Concatenate()([u1, c3])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)

    u2 = UpSampling2D((2, 2))(c4)
    concat2 = Concatenate()([u2, c2])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat2)

    u3 = UpSampling2D((2, 2))(c5)
    concat3 = Concatenate()([u3, c1])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same', name='last_conv')(concat3)

    # Classifier
    gap = GlobalAveragePooling2D()(c6)
    dropout = Dropout(0.5)(gap)
    output = Dense(num_classes, activation='softmax', name='dense_2')(dropout)

    model = Model(inputs=inputs, outputs=output)
    return model
