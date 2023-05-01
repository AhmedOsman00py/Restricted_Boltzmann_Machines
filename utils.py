import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

from sklearn.manifold import TSNE

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import torch


# --- Pytorch model backup
def save_model(model, path, is_torch=True):
    """save the model in a .pth file"""

    file_path = f"save/model/{path}"

    if is_torch:
        torch.save(model.state_dict(), file_path)
    else:
        pickle.dump(model, open(file_path, "wb"))


# --- Loss plots
def plot_loss(num_epochs, loss_, loss_epochs):
    """plot the loss while training"""

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    # Loss per Steps
    ax[0].plot(loss_)
    ax[0].set_xlabel('Steps for all epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss for each Step')

    # Loss per Epoch
    ax[1].plot(range(1, 10 + 1), loss_epochs)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss for each Epoch')

    plt.show()


# --- RBM weights
def plot_weights(weights):
    """display the weights"""

    image_dims = int(np.sqrt(weights.shape[1]))
    num_images = int(np.sqrt(weights.shape[0]))

    fig, axes = plt.subplots(num_images, num_images, figsize=(10, 10))
    for i in range(num_images):
        for j in range(num_images):
            index = i * num_images + j
            axes[i][j].imshow(
                weights[index].reshape(image_dims, image_dims), cmap="gray"
            )
            axes[i][j].axis("off")

    plt.show()


# --- RBM generated images
def plot_save_image(file_path, real, generated):
    """display and save the generated images"""

    np_real = np.transpose(real.numpy(), (1, 2, 0))
    np_generated = np.transpose(generated.numpy(), (1, 2, 0))

    f = f"save/image/{file_path}.png"

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    plt.imshow(np_real)
    plt.title("Real images")

    ax = fig.add_subplot(122)
    plt.imshow(np_generated)
    plt.title("Generated images")

    plt.savefig(f)
    plt.show()


# t-SNE (2D visualization of the dataset)
def plot_tsne(X, y):
    """plot the t-SNE of the dataset"""

    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
    df_tsne = pd.DataFrame(
        {
            "First component": X_tsne[:, 0],
            "Second component": X_tsne[:, 1],
            "Digit": y,
        }
    )
    df_tsne = df_tsne.sort_values(by="Digit")

    sns.scatterplot(
        data=df_tsne,
        x="First component",
        y="Second component",
        hue="Digit",
        palette="bright",
    )
    plt.show()


# --- Model evaluation
def evaluate_model(X_train, X_test, y_train, y_test, model, name):
    """evaluate some metrics of a given ML model"""

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_,
    )
    disp.plot()
    plt.title(f"{name} Confusion matrix")
    plt.figure(figsize=(7, 4))

    file_name = name.lower().replace(" ", "_")
    save_model(model, f"{file_name}.pkl", False)

    plot_tsne(X_test, y_pred)
