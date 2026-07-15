import sys
import os
import subprocess

# =====================================================================
# SELF-CONTAINED DEPENDENCY INSTALLATION & AUTO-REBOOT ENGINE
# =====================================================================
def auto_setup_environment():
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
        import matplotlib
        import PIL
        import numpy as np
        import sklearn  
        import pandas as pd
        import importlib_resources 
        return 
    except ImportError:
        print("\n[SETUP] Missing or uncached dependencies detected. Initiating installation...")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    dependencies = [
        "tensorflow[and-cuda]>=2.15.0",
        "tensorflow-datasets>=4.9.0",
        "matplotlib>=3.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",  
        "pandas>=2.0.0",      
        "importlib_resources"
    ]
    
    print(f"[SETUP] Enforcing specific library versions: {dependencies}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
    
    print("\n[SETUP] Environment successfully built! Re-executing script safely...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

auto_setup_environment()

# =====================================================================
# MAIN PIPELINE IMPORTS 
# =====================================================================
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

# =====================================================================
# SYSTEM SWITCHES CONFIGURATION MATRIX
# =====================================================================
CONFIG_MODE = "A100_1HR_RUN"  
HARDWARE_TARGET = "A5500"

MODE_MATRIX = {
    "DEBUG": {"final_epochs": 1},
    "STANDARD": {"final_epochs": 15},
    "A100_1HR_RUN": {"final_epochs": 50},
    "A100_3HR_RUN": {"final_epochs": 80}
}

HARDWARE_MATRIX = {
    "T4": {"mixed_precision": "mixed_float16", "tf32": False, "jit_compile": False, "suggested_batch": 32},
    "A5500": {"mixed_precision": "mixed_bfloat16", "tf32": True, "jit_compile": True, "suggested_batch": 64},
    "A100": {"mixed_precision": "mixed_bfloat16", "tf32": True, "jit_compile": True, "suggested_batch": 256}
}

PARAMS = MODE_MATRIX[CONFIG_MODE]
HW_PROFILE = HARDWARE_MATRIX[HARDWARE_TARGET]

IMAGE_SIZE = (96, 96)
NUM_CLASSES = 10
OUTPUT_DIR = "./mobilenet_direct_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if HW_PROFILE["tf32"]:
    tf.config.experimental.enable_tensor_float_32_execution(True)

tf.keras.mixed_precision.set_global_policy(HW_PROFILE["mixed_precision"])

# =====================================================================
# DATA STREAMING ENGINE
# =====================================================================
print("\n[DATA] Fetching STL-10 dataset...")
[train_raw, val_raw], ds_info = tfds.load('stl10', split=['train', 'test'], as_supervised=True, with_info=True)

class_names = ds_info.features['label'].names

def preprocess_data(image, label):
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = (train_raw
                 .map(preprocess_data, num_parallel_calls=AUTOTUNE)
                 .shuffle(5000)
                 .batch(HW_PROFILE["suggested_batch"])
                 .prefetch(AUTOTUNE))

val_dataset = (val_raw
               .map(preprocess_data, num_parallel_calls=AUTOTUNE)
               .batch(HW_PROFILE["suggested_batch"])
               .prefetch(AUTOTUNE))

# =====================================================================
# MODEL ARCHITECTURE DEPLOYMENT
# =====================================================================
print("\n=== BUILDING DEDICATED MOBILENETV2 PIPELINE ===")
input_shape = (*IMAGE_SIZE, 3)

inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
x = tf.keras.layers.RandomRotation(0.15)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

base_network = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

base_network.trainable = True
total_layers = len(base_network.layers)
for layer in base_network.layers[:-int(total_layers * 0.3)]:
    layer.trainable = False

x = base_network(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(units=256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'], jit_compile=HW_PROFILE["jit_compile"])

# =====================================================================
# PRODUCTION TRAINING EXECUTION LOOP
# =====================================================================
print(f"\n=== TRAINING MOBILENETV2 DIRECTLY FOR {PARAMS['final_epochs']} MAX EPOCHS ===")
optimization_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
]

history = model.fit(train_dataset, validation_data=val_dataset, epochs=PARAMS["final_epochs"], callbacks=optimization_callbacks, verbose=1)
model.save(os.path.join(OUTPUT_DIR, "direct_mobilenet_model.keras"))

# =====================================================================
# PROFILING ENGINE (INFRASTRUCTURE SPEED & PARAMETERS)
# =====================================================================
print("\n[BENCHMARK] Calculating footprint and tracking inference latency...")

total_params = model.count_params()
memory_footprint_mb = (total_params * 4) / (1024 * 1024)

for img_batch, _ in val_dataset.take(1):
    _ = model.predict(img_batch, verbose=0)

inference_latencies = []
total_images_evaluated = 0

for img_batch, _ in val_dataset.take(5):
    start_time = time.time()
    _ = model.predict(img_batch, verbose=0)
    end_time = time.time()
    inference_latencies.append(end_time - start_time)
    total_images_evaluated += img_batch.shape[0]

avg_inference_speed_ms = (sum(inference_latencies) / total_images_evaluated) * 1000
final_val_acc = history.history['val_accuracy'][-1] * 100

# =====================================================================
# DYNAMIC REPORT FORMATTING ENGINE
# =====================================================================
report_data = {
    "Model Type": ["Advanced (5-Layer Fine-Tuned + CutMix)"],
    "Total Parameters": [f"{total_params:,}"],
    "Memory Footprint": [f"{memory_footprint_mb:.2f} MB"],
    "Inference Speed (per image)": [f"{avg_inference_speed_ms:.3f} ms"],
    "Val Accuracy": [f"{final_val_acc:.2f}%"]
}

df_report = pd.DataFrame(report_data)

print("\n" + "="*80)
print("FINAL PROJECT ENGINEERING REPORT: THE FARM AI")
print("="*80)
print(df_report.to_string())
print("="*80)

df_report.to_csv(os.path.join(OUTPUT_DIR, "engineering_report_row.csv"), index=False)

# =====================================================================
# DIAGNOSTICS & TELEMETRY GENERATION
# =====================================================================
raw_predictions = model.predict(val_dataset, verbose=0)
y_pred = np.argmax(raw_predictions, axis=1)

y_true = []
for _, labels in val_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)

text_report = classification_report(y_true, y_pred, target_names=class_names)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(text_report)

fig, ax = plt.subplots(figsize=(10, 9))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, ax=ax, cmap='Blues', xticks_rotation='vertical')
plt.title('MobileNetV2 Direct Run - Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

print("\n=== PIPELINE RUN COMPLETED ===")