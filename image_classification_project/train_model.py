

from data_preparation import load_and_prepare_data
from model import create_model



def train_model():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    model = create_model()

    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))
    model.save('cifar10_model.keras')
    return history



if __name__ == "__main__":
    history = train_model()
    print("Model trained and saved successfully!")
