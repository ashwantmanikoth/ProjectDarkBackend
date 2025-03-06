# filepath: /Users/ashwantmanikoth/Desktop/programming/ProjectDark/Backend/src/utils/qdrant_operations.py
import qdrant_client
from qdrant_client.http import models
import torch, os
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

client = qdrant_client.QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "project_dark"
colpali_model = ColPali.from_pretrained("vidore/colpali-v1.2", torch_dtype=torch.bfloat16, device_map="cpu")
colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

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

def collection_exists():
    collections = client.get_collections()
    return any(c.name == COLLECTION_NAME for c in collections.collections)

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