

from keras.models import load_model
from data_preparation import load_and_prepare_data


def evaluate_model():
    _, _, X_test, _, _, y_test = load_and_prepare_data()
    model = load_model('cifar10_model.h5')

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy:.2f}')



if __name__ == "__main__":
    evaluate_model()
