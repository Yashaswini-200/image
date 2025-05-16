from backend.predictionFunction import predict_image
if __name__ == "__main__":
    # Manually test the backend with an image
    image_path = "backend/training_data/ai_folder/ai_folder/7e34225424b78a952f0a3d160b.jpg"  # Replace with a real image path
    try:
        result = predict_image(image_path)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")
