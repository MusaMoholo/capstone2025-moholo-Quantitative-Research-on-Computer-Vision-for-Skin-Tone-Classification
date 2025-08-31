# src/augment.py
import tensorflow as tf

def static_augment(x):
    # Horizontal flip
    x = tf.image.random_flip_left_right(x)
    # Brightness jitter (±0.1 in [0,1] space)
    x = tf.image.random_brightness(x, max_delta=0.1)
    # Contrast jitter (~±10%)
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
    # Light gaussian noise
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.02, dtype=x.dtype)
    x = x + noise
    # Keep within [0,1]
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x
