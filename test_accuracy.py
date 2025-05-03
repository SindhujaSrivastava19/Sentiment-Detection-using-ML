from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = load_model('Emotion_Detection.h5')

# Corrected path
validation_data_dir = r'C:/Users/91637/Desktop/Extraa/PROJECTS/Emotion Detection/Dataset/test'

# Data generator
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
