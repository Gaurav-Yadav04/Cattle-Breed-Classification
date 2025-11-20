import gradio as gr
import tensorflow as tf
import numpy as np
import cv2


# CONFIGURATION

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Gir', 'Jersey cattle', 'Sahiwal']


# LOAD TFLITE MODEL

try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    raise RuntimeError(f"Error loading TFLite model: {e}")


# PREPROCESSING FUNCTION

def preprocess_image(image):
    """Convert image to RGB, resize and normalize"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# PREDICTION FUNCTION

def predict(image):
    if image is None:
        return {"Error": 0.0}

    try:
        img = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(preds))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(preds[predicted_index])

        # Return result
        
        return f"{predicted_class} ({confidence*100:.2f}%)"

    except Exception as e:
        return {"Error during prediction": str(e)}


# GRADIO INTERFACE

title = "Cattle Breed Classifier"
description = """
Upload an image of a cow to identify whether it's **Gir**, **Jersey cattle**, or **Sahiwal** 🐮.  
This app uses a fine-tuned **MobileNetV2** model (TFLite version) for fast and accurate predictions.
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Cattle Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
