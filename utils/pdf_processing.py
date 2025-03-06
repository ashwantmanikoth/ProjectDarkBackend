# filepath: /Users/ashwantmanikoth/Desktop/programming/ProjectDark/Backend/src/utils/pdf_processing.py
from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_path, doc_id, output_folder="images"):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)
    
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_folder, f"{doc_id}_page_{i}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    
    return image_paths