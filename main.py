import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained CNN model from the .h5 extension file
model = tf.keras.models.load_model('emotion_model.h5')

# Define a dictionary to map emotion indices to emotion labels
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Function to preprocess the image for the CNN model
def preprocess_image(image):
    # Resize the image to 48x48 pixels
    resized_image = cv2.resize(image, (48, 48))

    # Convert the image to grayscale (single channel)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = gray_image / 255.0

    # Add a channel dimension to match the input shape of the CNN model
    input_image = np.expand_dims(normalized_image, axis=-1)

    return input_image

# Function to predict the user's emotion using the CNN model
def predict_emotion(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Convert the image to a batch format for model prediction
    input_batch = np.expand_dims(preprocessed_image, axis=0)

    # Perform inference using the CNN model
    predictions = model.predict(input_batch)

    # Get the emotion index with the highest confidence
    emotion_index = np.argmax(predictions[0])

    # Get the emotion label from the index
    predicted_emotion = emotion_labels[emotion_index]

    return predicted_emotion

# Main function for live video emotion detection
def main():
    # Use OpenCV to access the user's camera
    camera = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()

        # Perform emotion detection on the frame
        emotion = predict_emotion(frame)

        # Display the frame with the predicted emotion label
        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Emotion Detection', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
