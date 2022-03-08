import json
from inference_onnx import ColaONNXPredictor

inferencing_instance = ColaONNXPredictor(model_path="./models/model.onnx")


def lambda_handler(event, context):
    """
    Lambda function handler for predicting linguistic acceptability of the given sentence

    Args:
        event (_type_): _description_
        context (_type_): _description_
    """

    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Input received: {body['sentence']}")
        return{
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response)
        }
    else:
        return inferencing_instance.predict(event["sentence"])
    
if __name__ == "__main__":
    test = {"sentence": "I like to play violin"}
    print(lambda_handler(test, None))