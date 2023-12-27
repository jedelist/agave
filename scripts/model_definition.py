from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer

def create_colorization_model(input_shape):
    model = Sequential()

    # Input layer
    model.add(InputLayer(input_shape=input_shape))

    # Encoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    # additional encoder layers if needed to enhance the model

    # Decoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))         #bring the image back to 256x256
    # additional decoder 

    # Output layer with 3 channels for RGB output
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))      #Maybe use tanh
    #model.add(UpSampling2D((2, 2)))

    # Compile the model with loss function (mse)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    input_shape = (256, 256, 1)  # Grayscale image shape for this project (1 channel)
    model = create_colorization_model(input_shape)
    model.summary()  # Print summary of the model -- use to check for bugs