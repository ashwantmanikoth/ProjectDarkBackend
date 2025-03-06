# filepath: /Users/ashwantmanikoth/Desktop/programming/ProjectDark/Backend/src/routes.py
from flask import request, jsonify
import os
from utils.pdf_processing import convert_pdf_to_images
from utils.qdrant_operations import (
    create_collection,
    collection_exists,
    store_images_in_qdrant,
    retrieve_relevant_pages,
)
from utils.colpali_operations import generate_doc_id, send_images_to_model
from utils.error_handling import handle_exception


def configure_routes(app):
    @app.route("/upload", methods=["POST"])
    def upload_pdf():
        try:
            print("Received PDF file")
            file = request.files["file"]
            pdf_path = f"uploads/{file.filename}"
            os.makedirs("uploads", exist_ok=True)
            file.save(pdf_path)
            print("PDF file saved")

            doc_id = generate_doc_id(file.filename)
            image_paths = convert_pdf_to_images(pdf_path, doc_id)
            print("PDF converted to images")

            if not collection_exists():
                create_collection()

            store_images_in_qdrant(image_paths, doc_id=doc_id)

            return jsonify({"message": "PDF processed successfully!", "doc_id": doc_id})
        except Exception as e:
            return handle_exception(e)

    @app.route("/query", methods=["POST"])
    def query_documents():
        try:
            print("Received query")
            query_text = request.json["query"]
            matched_images = retrieve_relevant_pages(query_text)
            answer = send_images_to_model(matched_images, query_text)
            return jsonify({"answer": answer})
        except Exception as e:
            return handle_exception(e)
