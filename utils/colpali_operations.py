# filepath: /Users/ashwantmanikoth/Desktop/programming/ProjectDark/Backend/src/utils/colpali_operations.py
import uuid
import boto3
from PIL import Image


def generate_doc_id(filename):
    return int(uuid.uuid4().int >> 96)


def read_and_encode_image(image_path: str):
    print("Image path is", image_path)
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    image_format = Image.open(image_path).format.lower()
    message_content = {
        "image": {
            "format": image_format,
            "source": {"bytes": image_bytes},
        }
    }
    return message_content


def send_images_to_model(matched_images, query):
    system_prompt = "You are a helpful assistant for question answering. Given the context, answer the question strictly from context. If the content is not adequate, reply with From the given context, I can not answer the question."
    model_id = "amazon.nova-pro-v1:0"

    content_list = [read_and_encode_image(img) for img in matched_images]
    content_list.append({"text": query})
    system = [{"text": system_prompt}]
    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]
    inf_params = {"temperature": 0.3, "maxTokens": 5000}
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId=model_id, messages=messages, system=system, inferenceConfig=inf_params
    )
    return response["output"]["message"]["content"][0]["text"]
