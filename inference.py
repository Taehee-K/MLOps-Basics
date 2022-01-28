import torch
from data import DataModule
from model import ColaModel


class ColaPredictor:
    """Cola Predictor Inference Module
    
    Examples
    --------
    >>> import ColaPredictor
    >>> model_path = "./models/epoch=4-step=1339.ckpt"
    >>> sentence = "There is a boy sitting on a bench"
    >>> predictor = ColaPredictor(model_path)
    >>> print(predictor.predict(sentence))
    [{'label': 'unacceptable', 'score': 0.34601861238479614}, {'label': 'acceptable', 'score': 0.6539814472198486}]
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["unacceptable", "acceptable"]

    def predict(self, text: str):
        inference_sample = {"sentence": text}
        proceed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([proceed["input_ids"]]),
            torch.tensor([proceed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = input("Input Text: ")
    predictor = ColaPredictor("./models/epoch=4-step=1339.ckpt")  
    print(predictor.predict(sentence))