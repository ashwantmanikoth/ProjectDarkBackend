from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_path
from colpali_engine.models import ColPali, ColPaliProcessor
import torch, os, boto3, qdrant_client
from qdrant_client.http import models
from PIL import Image
import uuid

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes
client = qdrant_client.QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "project_dark"

colpali_model = ColPali.from_pretrained("vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map="cpu")
colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

def generate_doc_id(filename):
    return int(uuid.uuid4().int >> 96)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    print("Received PDF file")
    IMAGE_DIR = "images"
    file = request.files["file"]
    pdf_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(pdf_path)
    print("PDF file saved")
    
    doc_id = generate_doc_id(file.filename)

    image_paths = convert_pdf_to_images(pdf_path, doc_id)

    print("PDF converted to images")
    
    if not collection_exists(COLLECTION_NAME):
        create_collection()
    
    store_images_in_qdrant(image_paths, doc_id=doc_id)
    
    return jsonify({"message": "PDF processed successfully!", "doc_id": doc_id})

def create_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        on_disk_payload=True,
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            on_disk=True,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    )
    print("Collection created")

def collection_exists(collection_name):
    collections = client.get_collections()
    return any(c.name == collection_name for c in collections.collections)

def convert_pdf_to_images(pdf_path, doc_id, output_folder="images"):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)
    
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f"{doc_id}_page_{i}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    
    return image_paths

def store_images_in_qdrant(image_paths, doc_id):
    try:
        dataset = [{"doc_id": doc_id, "page_num": i, "image": path} for i, path in enumerate(image_paths)]
        print("Colpali processing starts")

        with torch.no_grad():
            batch_images = colpali_processor.process_images([Image.open(item["image"]) for item in dataset]).to(colpali_model.device)
            embeddings = colpali_model(**batch_images)
        
        print("Colpali processing completed")

        points = [
            models.PointStruct(
                id=doc_id * 1000 + i,
                vector=embedding.tolist(),
                payload={"doc_id": doc_id, "page_num": i}
            ) for i, embedding in enumerate(embeddings)
        ]
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("Images stored in Qdrant")
        
    except Exception as ex:
        print(ex)
        print("Error storing images in Qdrant")

        
@app.route("/query", methods=["POST"])
def query_documents():
    print("Received query")
    query_text = request.json["query"]
    matched_images = retrieve_relevant_pages(query_text)
    answer = send_images_to_model(matched_images, query_text)
    return jsonify({"answer": answer})

def retrieve_relevant_pages(query_text, output_folder="images"):
    with torch.no_grad():
        text_embedding = colpali_processor.process_queries([query_text]).to(colpali_model.device)
        text_embedding = colpali_model(**text_embedding)
    
    query_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=text_embedding[0].cpu().float().numpy().tolist(),
        limit=5
    )
    
    matched_images = [os.path.join(f"images/{res.payload['doc_id']}_page_{res.payload['page_num']}.png") for res in query_result.points]
    
    return matched_images

def read_and_encode_image(image_path: str):
    
    print("Image path is",image_path)

    with open(image_path, 'rb') as image_file:
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
    system_prompt = 'You are a helpful assistant for question answering. Given the context, answer the question strictly from context. If the content is not adequate, reply with From the given context, I can not answer the question.'
    model_id = 'amazon.nova-pro-v1:0'
    
    content_list = [read_and_encode_image(img) for img in matched_images]

    content_list.append({"text": query})

    system = [{"text": system_prompt}]
    
    messages = [{
        "role": "user",
        "content": content_list,
    }]
    
    inf_params = {"temperature": .3, "maxTokens": 5000}

    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=system,
        inferenceConfig=inf_params
    )
    
    return response["output"]["message"]["content"][0]["text"]

if __name__ == "__main__":
    app.run(debug=True, port=5000)