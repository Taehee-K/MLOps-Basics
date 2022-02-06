import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaONNXPredictor:
    """Cola Predictor Inference Module with ONNX Runtime

    Examples
    --------
    >>> import ColaONNXPredictor
    >>> model_path = "./models/model.onnx"
    >>> sentence = "There is a boy sitting on a bench"
    >>> predictor = ColaONNXPredictor(model_path)
    >>> print(predictor.predict(sentence))
    function:'predict' took: 0.00588 sec
    [{'label': 'unacceptable', 'score': 0.39150003}, {'label': 'acceptable', 'score': 0.60849994}]
    """

    def __init__(self, model_path: str) -> None:
        # create inference session -> load onnx model
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text: str) -> None:
        inference_sample = {"sentence": text}
        proceed = self.processor.tokenize_data(inference_sample)
        # preparing inputs
        ort_inputs = {
            "input_ids": np.expand_dims(proceed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(proceed["attention_mask"], axis=0),
        }

        # run inference (None = get all outputs)
        ort_output = self.ort_session.run(None, ort_inputs)

        # normalising outputs
        scores = softmax(ort_output[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    model_path = "./models/model.onnx"
    sentence = input("Input Text: ")
    predictor = ColaONNXPredictor(model_path)
    print(predictor.predict(sentence))
