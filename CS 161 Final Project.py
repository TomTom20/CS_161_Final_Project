"""CS 161 Final Project."""

import tensorflow as tf


def get_data():
    # Initialize MNIST Digit dataset.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return x_train, y_train, input_shape


def user():
    # Prompt user for if they want to rerun the program.
    validate = input("Would you like to run the program again? (y/n)")
    if validate == "y":
        return True
    else:
        return False


def read_setup(file):
    # Update and collect data from the setup.txt file.
    f = open(file, "r+")
    f.readlines[3] = "CURRENT PROGRAM STATUS: Running"
    f.write(f.readlines())
    f.close()
    return f.readlines()[1]


def init_setup(file):
    # Initialize the setup.txt file.
    f = open(file, "r+")
    f.readlines[1] = "###"
    f.readlines[3] = "CURRENT PROGRAM STATUS: Not Running"
    f.write(f.readlines())
    f.close()
    return None


def main(x_train, y_train, input_shape, num):
    # Run the data through the neural network.
    while True:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        try:
            model.fit(x=x_train, y=y_train, epochs=num)
        except TypeError:
            print("An invalid accuracy value was given in setup.txt.")
            print("Make sure to provide a value between 1 and 100.")
            print("Now defaulting to 10.")
            model.fit(x=x_train, y=y_train, epochs=10)
        user()

    init_setup("setup.txt")


main(get_data(), read_setup("setup.txt"))
