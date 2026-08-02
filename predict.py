import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("model/cnn_model.keras")

# -----------------------------
# CIFAR-10 Class Names
# -----------------------------
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(img_path):
    """
    Predict image class using trained CNN model.

    Parameters
    ----------
    img_path : str
        Path of uploaded image.

    Returns
    -------
    predicted_class : str
    confidence : float
    top3_predictions : list
    """

    # Load image
    img = image.load_img(img_path, target_size=(32, 32))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize image
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Best prediction
    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(
        round(float(predictions[predicted_index]) * 100, 2)
    )

    # Top-3 Predictions
    top3_indices = predictions.argsort()[-3:][::-1]

    top3_predictions = []

    for index in top3_indices:

        top3_predictions.append(
            {
                "class": class_names[int(index)],
                "confidence": float(
                    round(float(predictions[index]) * 100, 2)
                )
            }
        )

    return predicted_class, confidence, top3_predictions


# -----------------------------
# Test Prediction
# -----------------------------
if __name__ == "__main__":

    image_path = "test.jpg"

    predicted_class, confidence, top3 = predict_image(image_path)

    print("\nPrediction")
    print("----------------------------")
    print(f"Class      : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nTop 3 Predictions")

    for item in top3:
        print(
            f"{item['class']} : {item['confidence']:.2f}%"
        )