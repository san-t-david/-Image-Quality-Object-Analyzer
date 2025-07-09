import streamlit as st
from PIL import Image
import numpy as np
import base64
import os

# Optional: Load YOLOv8 model for object detection
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')  # Pretrained nano model
    yolo_available = True
except:
    yolo_available = False

# Function to detect objects
def detect_objects(image):
    results = model(image)
    labels = results[0].boxes.cls.cpu().numpy()
    return results[0].plot(), model.names, labels

# Function to analyze image quality
def analyze_image_quality(image):
    img_np = np.array(image)
    brightness = np.mean(img_np)
    clarity_score = np.std(img_np)
    focus_level = clarity_score

    feedback = []
    if brightness < 50:
        feedback.append("Low lighting")
    if clarity_score < 10:
        feedback.append("Blurry image")
    if len(feedback) == 0:
        feedback.append("Image looks clear and well-lit")

    return brightness, focus_level, feedback

# Generate download link
def create_download_link(text, filename="report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📄 Download Report</a>'

# UI Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506765515384-028b60a970df");
        background-size: cover;
        background-attachment: fixed;
    }
    .title-text {
        color: white;
        text-shadow: 2px 2px 4px black;
        font-size: 3em;
        text-align: center;
    }
    .stMarkdown {
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    .object-label {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown('<h1 class="title-text">📸 Image Quality & Object Analyzer</h1>', unsafe_allow_html=True)
st.write("Upload an image to analyze its clarity, brightness, and focus. AI object detection enabled if available.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 Uploaded Image', use_column_width=True)

    # Quality Analysis
    brightness, focus, feedback = analyze_image_quality(image)
    st.subheader("🔍 Quality Analysis")
    st.write(f"- Brightness: {brightness:.2f}")
    st.write(f"- Focus Level (Sharpness): {focus:.2f}")
    st.write(f"- Feedback: {', '.join(feedback)}")

    # Object Detection (if available)
    if yolo_available:
        st.subheader("🧠 Object Detection")
        output_image, names, labels = detect_objects(image)
        st.image(output_image, caption="Detected Objects", use_column_width=True)
        detected = [names[int(lbl)] for lbl in labels]

        if detected:
            st.write("Detected:")
            for obj in sorted(set(detected)):
                st.markdown(f'<div class="object-label">🔹 {obj}: {detected.count(obj)}</div>', unsafe_allow_html=True)
        else:
            st.warning("No objects detected.")
    else:
        st.warning("YOLOv8 not installed. Object detection is disabled.")

    # Report Download
    report = f"Brightness: {brightness:.2f}\nFocus: {focus:.2f}\nFeedback: {', '.join(feedback)}"
    if yolo_available:
        object_summary = "\n".join(f"{obj}: {detected.count(obj)}" for obj in set(detected))
        report += f"\n\nDetected Objects:\n{object_summary}"

    st.markdown(create_download_link(report), unsafe_allow_html=True)
