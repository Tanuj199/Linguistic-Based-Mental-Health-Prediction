from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Project Configuration ---
MODEL_FILE = 'best_model.h5'

# NLP Parameters
VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 10
BATCH_SIZE = 32

# --- Check for pre-trained model ---
if os.path.exists(MODEL_FILE):
    print("Pre-trained model found. Skipping training.")
    # You can load the model here if needed, but for training, we'll skip
    # this step to avoid accidental overwrites.
    exit()

# --- Load preprocessed data ---
if not os.path.exists('processed_data.npz'):
    print("Error: 'processed_data.npz' not found.")
    print("Please run the data preprocessing script first.")
    exit()

data = np.load('processed_data.npz', allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']

print("Data loaded successfully.")

# --- Build the Model ---
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(units=LSTM_UNITS, return_sequences=False),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train the Model ---
print("\nStarting model training...")

# Callbacks to save the best model and stop training early
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

print("\nModel training complete.")
print(f"Best model saved to '{MODEL_FILE}'.")
