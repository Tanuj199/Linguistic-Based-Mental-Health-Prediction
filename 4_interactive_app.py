import gradio as gr

# Parameters
MAX_LEN = 100

# Check if model and tokenizer files exist
if not os.path.exists('best_model.h5') or not os.path.exists('tokenizer.pkl'):
    print("Error: Model or tokenizer files not found. Please run 1_data_preprocessing.py and 2_model_training.py first.")
    exit()

# Load the trained model and tokenizer
model = load_model('best_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

print("Model and tokenizer loaded for prediction.")

# --- Prediction Function ---
def predict_risk(text):
    """
    Predicts the mental health risk from a given text.
    """
    # Preprocess the input text
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#|[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return "Please enter some text.", "Neutral"

    # Tokenize and pad the sequence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Make the prediction
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    # Determine risk level and status based on a probability threshold
    if prediction >= 0.5:
        risk_label = "High Risk"
        risk_color = "red"
        message = "Based on the text, there are indicators of mental health risk. Please seek professional help if you or someone you know is struggling."
    else:
        risk_label = "Low Risk"
        risk_color = "green"
        message = "Based on the text, there are no strong indicators of mental health risk. However, this is not a substitute for a professional diagnosis."
    
    # Format the output for Gradio
    risk_output = f"<p style='font-size: 24px; color: {risk_color};'><b>Risk Level: {risk_label}</b></p>"
    
    return risk_output, message

# --- Gradio Interface ---
# Define the input and output components
input_text = gr.Textbox(lines=5, label="Enter text for mental health risk analysis")
output_risk = gr.HTML(label="Prediction Result")
output_message = gr.Textbox(label="Message", interactive=False)

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_risk,
    inputs=input_text,
    outputs=[output_risk, output_message],
    title="Linguistic-Based Mental Health Risk Prediction",
    description="This application analyzes text for linguistic patterns associated with mental health risk. **Note: This is a prototype and not a medical tool.**",
    theme=gr.themes.Soft()
)

# Launch the app
iface.launch(share=True)
