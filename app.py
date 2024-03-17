import gradio as gr
from transformers import pipeline

# Load the model
model_name = "knowledgator/comprehend_it-base"
classifier = pipeline("zero-shot-classification", model=model_name, device="cpu")

# Function to classify feedback
def classify_feedback(feedback_text):
    # Classify feedback using the loaded model
    labels = ["High Priority ticket", "Low Priority ticket", "Medium Priority ticket"]

    result = classifier(feedback_text, labels, multi_label=True)

    # Get the top label associated with the feedback
    top_label = result["labels"][0]

    return top_label

# Create Gradio interface
feedback_textbox = gr.Textbox(label="Enter your feedback:")
feedback_output = gr.Label(label="Top Label:")

gr.Interface(
    fn=classify_feedback,
    inputs=feedback_textbox,
    outputs=feedback_output,
    title="Feedback Classifier",
    description="Enter your feedback and get the priority label for your ticket."
).launch()
