import tensorflow as tf
from tensorflow.keras import layers, models

def build_mobilenetv3(num_classes=6, img_size=224, freeze_backbone=True, name=None):
    base = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        input_shape=(img_size, img_size, 3),
        weights="imagenet"
    )
    if freeze_backbone:
        base.trainable = False
    inp = layers.Input((img_size, img_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name=name or "mobilenetv3_large_head")

def compile_model(model, lr=3e-4, label_smoothing=0.0):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    num_classes = int(model.output_shape[-1])

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    if label_smoothing and float(label_smoothing) > 0.0:
        try:
            cce = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=float(label_smoothing),
                from_logits=False,
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            )
            def loss_fn(y_true, y_pred):
                y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
                return cce(y_true_oh, y_pred)
        except TypeError:
            eps = tf.convert_to_tensor(float(label_smoothing), dtype=tf.float32)
            invK = tf.constant(1.0/num_classes, dtype=tf.float32)
            def loss_fn(y_true, y_pred):
                y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
                y_true_sm = y_true_oh * (1.0 - eps) + eps * invK
                return tf.keras.backend.categorical_crossentropy(y_true_sm, y_pred, from_logits=False)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

