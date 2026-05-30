import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Nur GPU 1 verwenden
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[1], "GPU")
    tf.config.experimental.set_memory_growth(gpus[1], True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

train_ds = keras.utils.image_dataset_from_directory(
    "data/OCT/train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

val_ds = keras.utils.image_dataset_from_directory(
    "data/OCT/train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
)

# Klassen aus Ordnern verwenden ['CNV', 'DME', 'DRUSEN', 'NORMAL']
class_names = train_ds.class_names
print("Classes:", class_names)

# Dataset verkleinern (SMOKE TEST), 20 Batches Training, 5 Batches Validation
train_ds = train_ds.take(20)
val_ds = val_ds.take(5)

# InceptionV3 laden
base_model = keras.applications.InceptionV3(
    weights="imagenet", # Modell ist auf Imagenet vortrainiert
    include_top=False, # entfernt den alten Klassifikationskopf
    input_shape=(224, 224, 3),
)
base_model.trainable = False # Modell einfrieren

# Eigenes Modell bauen
model = keras.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(4, activation="softmax"),
])

# Modell vorbereiten
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=1)

model.save("models/smoke_tests/smoke_test_inceptionv3.keras")
print("Smoke test done.")
