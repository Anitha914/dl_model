import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

model = load_model('pretrainedmodel_OCT_feature_extraction.hdf5')
history = joblib.load('training_history_OCT_feature_extraction.pkl')

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history['accuracy'], 'bo-', label='Train Acc')
plt.plot(history['val_accuracy'], 'ro-', label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history['loss'], 'bo-', label='Train Loss')
plt.plot(history['val_loss'], 'ro-', label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Prepare Validation Data
val_dir = 'OCT_small/val'
val_datagen = ImageDataGenerator(preprocessing_function=None)  

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(299,299),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict on validation data
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
