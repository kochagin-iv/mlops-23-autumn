import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlxtend.plotting import plot_confusion_matrix
from model import make_model
from omegaconf import DictConfig
from sklearn.metrics import classification_report, confusion_matrix
from train import prepare_data


@hydra.main(config_path="configs", config_name="conf", version_base="1.3")
def main(config: DictConfig):
    _, _, X_test, _, _, y_test = prepare_data(
        config["path_train_dataset"], config["path_test_dataset"]
    )
    model, _, _ = make_model(X_test.shape[1])
    model.load_state_dict(torch.load(config["best_model_file"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    X_test_tensor = torch.Tensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        pred = np.argmax(outputs, axis=1)
        _, pred = torch.max(outputs.data, 1)

    _, y_true = torch.max(y_test.data, 1)

    CM = confusion_matrix(y_true, pred)

    _, _ = plot_confusion_matrix(conf_mat=CM, figsize=(10, 5))
    plt.savefig(f"{config['graphics_path']['test']}/confusion_matrix.jpg")

    print(classification_report(y_true, pred))

    d = {"Index": np.arange(y_test.shape[0]), "Activity": pred}
    final = pd.DataFrame(d)
    final.to_csv(f"{config['path_answers']}/human_activity_predictions.csv", index=False)


if __name__ == "__main__":
    main()
