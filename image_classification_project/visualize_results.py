import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from data_preparation import load_and_prepare_data



def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'.")



def plot_sample_predictions(X, y_true, y_pred, class_names):
    plt.figure(figsize=(12, 12))
    for i in range(16):
        idx = np.random.randint(0, len(X))
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[idx])
        plt.title(f"True: {class_names[np.argmax(y_true[idx])]}\nPred: {class_names[np.argmax(y_pred[idx])]}")
        plt.axis('off')
    plt.savefig('sample_predictions.png')
    print("Sample predictions plot saved as 'sample_predictions.png'.")




if __name__ == "__main__":

    from train_model import train_model
    history = train_model()
    plot_training_history(history)

    _, _, X_test, _, _, y_test = load_and_prepare_data()
    model = load_model('cifar10_model.keras')
    y_pred = model.predict(X_test)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plot_sample_predictions(X_test, y_test, y_pred, class_names)
