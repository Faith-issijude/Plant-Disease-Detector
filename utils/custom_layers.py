import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Reshape, UpSampling2D
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class PatchExtract(tf.keras.layers.Layer):
    def __init__(self, patch_size=2, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, x):
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dim = tf.shape(patches)[-1]
        patches = tf.reshape(patches, (tf.shape(x)[0], -1, patch_dim))
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size})
        return config

def mobilevit_block(x, patch_size=2, transformer_dim=64, num_heads=4, ff_dim=128):
    c = x.shape[-1]
    h, w = x.shape[1], x.shape[2]

    patches = PatchExtract(patch_size)(x)
    x_norm = LayerNormalization()(patches)
    att_out = MultiHeadAttention(num_heads=num_heads, key_dim=transformer_dim)(x_norm, x_norm)
    ff = Dense(ff_dim, activation='relu')(att_out)
    encoded = Dense(patch_size * patch_size * c)(ff)

    reshaped = Reshape((h // patch_size, w // patch_size, patch_size * patch_size * c))(encoded)
    upsampled = UpSampling2D(size=(patch_size, patch_size))(reshaped)

    return upsampled
