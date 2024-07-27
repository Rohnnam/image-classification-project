
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    print("Data prepared successfully!")
