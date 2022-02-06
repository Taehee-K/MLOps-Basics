import torch
from data import DataModule
from model import ColaModel
from utils import timing


class ColaPredictor:
    """Cola Predictor Inference Module

    Examples
    --------
    >>> import ColaPredictor
    >>> model_path = "./models/best_checkpoint.ckpt"
    >>> sentence = "There is a boy sitting on a bench"
    >>> predictor = ColaPredictor(model_path)
    >>> print(predictor.predict(sentence))
    function:'predict' took: 0.00738 sec
    [{'label': 'unacceptable', 'score': 0.3915000557899475}, {'label': 'acceptable', 'score': 0.6084999442100525}]
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=1)
        self.lables = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text: str) -> None:
        inference_sample = {"sentence": text}
        proceed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([proceed["input_ids"]]),
            torch.tensor([proceed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    model_path = "./models/best-checkpoint.ckpt"
    sentence = input("Input Text: ")
    predictor = ColaPredictor(model_path)
    print(predictor.predict(sentence))
