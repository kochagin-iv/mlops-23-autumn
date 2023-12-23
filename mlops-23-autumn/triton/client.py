from functools import lru_cache

import numpy as np
import pandas as pd
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_human_activity_predict(row):
    triton_client = get_client()
    input_row = InferInput(
        name="inputs", shape=row.shape, datatype=np_to_triton_dtype(np.float32)
    )
    input_row.set_data_from_numpy(row, binary_data=True)

    infer_output = InferRequestedOutput("predictions", binary_data=True)
    query_response = triton_client.infer(
        "onnx-human-activity", [input_row], outputs=[infer_output]
    )
    predictions = np.argmax(query_response.as_numpy("predictions")[0])
    return predictions


def main():
    labels = {
        0: "STANDING",
        1: "SITTING",
        2: "LAYING",
        3: "WALKING",
        4: "WALKING_DOWNSTAIRS",
        5: "WALKING_UPSTAIRS",
    }

    tests = [
        {
            "filepath": "../datasets/human_activity_predictions/train.csv",
            "row": 1,
            "answer": "STANDING",
        },
        {
            "filepath": "../datasets/human_activity_predictions/train.csv",
            "row": 151,
            "answer": "WALKING_UPSTAIRS",
        },
        {
            "filepath": "../datasets/human_activity_predictions/train.csv",
            "row": 297,
            "answer": "WALKING_DOWNSTAIRS",
        },
        {
            "filepath": "../datasets/human_activity_predictions/train.csv",
            "row": 379,
            "answer": "SITTING",
        },
        {
            "filepath": "../datasets/human_activity_predictions/train.csv",
            "row": 52,
            "answer": "LAYING",
        },
    ]
    for test in tests:
        row = pd.read_csv(test["filepath"], skiprows=test["row"], nrows=1, header=None)
        # row = row.drop(columns=["Activity"])
        row = np.array(row.iloc[:, :-1].iloc[:1], dtype="f")
        output = call_triton_human_activity_predict(row)
        print(f"Label for row is {test['answer']}, predicted label is {labels[output]}")


if __name__ == "__main__":
    main()
