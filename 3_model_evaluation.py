import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Check if the model and data files exist
if not os.path.exists('best_model.h5'):
    print("Error: 'best_model.h5' not found. Please run 2_model_training.py first.")
    exit()

if not os.path.exists('processed_data.npz'):
    print("Error: 'processed_data.npz' not found. Please run 1_data_preprocessing.py first.")
    exit()

# Load the trained model
model = load_model('best_model.h5')
print("Model loaded successfully.")

# Load the test data
data = np.load('processed_data.npz')
X_test, y_test = data['X_test'], data['y_test']
print("Test data loaded successfully.")

# Make predictions on the test set
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int)

# --- Evaluation Metrics ---
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Not Depressed', 'Depressed']))

# --- Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot the confusion matrix for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Depressed', 'Depressed'], yticklabels=['Not Depressed', 'Depressed'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
