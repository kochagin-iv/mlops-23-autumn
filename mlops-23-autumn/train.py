import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from model import make_model
from omegaconf import DictConfig
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data(train_file, test_file):
    df = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    scaler = MinMaxScaler()
    scaler.fit(df.iloc[:, 0:562])
    mat_train = scaler.transform(df.iloc[:, 0:562])

    scaler = MinMaxScaler()
    scaler.fit(test.iloc[:, 0:562])
    mat_test = scaler.transform(test.iloc[:, 0:562])

    activity_mapping = {activity: i for i, activity in enumerate(df["Activity"].unique())}
    # print(activity_mapping)
    df["n_Activity"] = df.Activity.map(activity_mapping)

    activity_mapping = {
        activity: i for i, activity in enumerate(test["Activity"].unique())
    }
    test["n_Activity"] = test.Activity.map(activity_mapping)

    df.drop(["Activity"], axis=1, inplace=True)
    test.drop(["Activity"], axis=1, inplace=True)

    y_train = F.one_hot(torch.tensor(df.n_Activity.values), num_classes=6)
    y_test = F.one_hot(torch.tensor(test.n_Activity.values), num_classes=6)

    X_train = mat_train
    X_test = mat_test

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.33, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@hydra.main(config_path="configs", config_name="conf", version_base="1.3")
def main(config: DictConfig):
    X_train, X_val, _, y_train, y_val, _ = prepare_data(
        config["path_train_dataset"], config["path_test_dataset"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.Tensor(X_train).to(device)
    y_train_tensor = torch.Tensor(y_train).to(device)

    X_val_tensor = torch.Tensor(X_val).to(device)
    y_val_tensor = torch.Tensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model, criterion, optimizer = make_model(X_train.shape[1])
    model.to(device)

    history = {}
    history["val"] = {}
    history["val"]["loss"] = []
    history["val"]["accuracy"] = []

    history["train"] = {}
    history["train"]["loss"] = []
    history["train"]["accuracy"] = []

    best_val_accuracy = 0.0

    lr_reduce = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=1, verbose=True, min_lr=0.0001
    )

    for epoch in range(int(config["number_epochs"])):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total

        history["train"]["accuracy"].append(train_accuracy)
        history["train"]["loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels.float())

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)

                _, labels = torch.max(labels, 1)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total

        history["val"]["loss"].append(val_loss)
        history["val"]["accuracy"].append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{int(config['number_epochs'])} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%"
        )

        lr_reduce.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config["best_model_file"])

    rcParams["figure.figsize"] = 10, 4
    plt.plot(history["train"]["accuracy"])
    plt.plot(history["val"]["accuracy"])

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"{config['graphics_path']['train']}/model_accuracy.jpg")
    plt.close()
    # summarize history for loss
    plt.plot(history["train"]["loss"])
    plt.plot(history["val"]["loss"])

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"{config['graphics_path']['train']}/model_loss.jpg")

    dummy_input_batch = next(iter(val_loader))[0]
    dummy_input = torch.unsqueeze(dummy_input_batch[0], 0)

    model_path = "mlops-23-autumn/datasets/human_activity_predictions"
    model_name = "model"
    torch.onnx.export(
        model,
        dummy_input,
        model_path + "_" + model_name + ".onnx",
        export_params=True,
        input_names=["inputs"],
        output_names=["predictions"],
        dynamic_axes={
            "inputs": {0: "BATCH_SIZE"},
            "predictions": {0: "BATCH_SIZE"},
        },
    )


if __name__ == "__main__":
    main()
