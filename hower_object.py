import torch
import clip
import os
import json
from PIL import Image
from ultralytics import YOLO

# Load the YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def detect_objects(image_path):
    try:
        results = yolo_model(image_path)
        # results[0].show()
        return results
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return None

def crop_object(image, coordinates):
    x1, y1, x2, y2 = coordinates
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def encode_image(image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_tensor)
    return image_feature

def find_similar_images(query_feature, dataset_features, top_k=5):
    similarities = torch.nn.functional.cosine_similarity(query_feature, dataset_features, dim=-1)
    values, indices = similarities.topk(top_k)
    return indices, values

def hower_image_similarity(image_path):
    # Load and encode dataset images
    products = [] 
    dataset_folder = 'dataset'
    metadata_file = os.path.join(dataset_folder, 'metadata.json')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    dataset_image_features = []
    for item in metadata:
        img_path = os.path.join(dataset_folder, 'images', item['filename'])
        dataset_image = Image.open(img_path).convert("RGB")
        dataset_image_features.append(encode_image(dataset_image))

    dataset_image_features = torch.cat(dataset_image_features, dim=0).to(device)
    # image_feature = encode_image(image_path)
    # indices, values = find_similar_images(image_feature, dataset_image_features)
    

            # Print the top matches for the current object
    # print("Top matches:")
    # for idx, value in zip(indices, values):
    #     match = metadata[idx]
    #     print(f"Match: {match['product_name']} with similarity score {value.item():.2f}, URL: {match['product_url']}")
    # print()  # Print a blank line for separation
    # # Process the input image
    results = detect_objects(image_path)

    # Handle multiple detections or different structure in results
    if isinstance(results, list):
        results = results[0]  # Take the first detection if there are multiple
    if results and results.boxes.xyxy.shape[0] > 0:
        boxes = results.boxes.xyxy  # Accessing bounding boxes

        for i, box in enumerate(boxes):
            class_index = int(box[-1].item())
            if class_index in results.names:
                class_label = results.names[class_index]
            else:
                class_label = 'Unknown'

            # Print detected label for current object
            print(f"Detected Label {i+1}: {class_label}")

            # Access bounding box coordinates
            x1, y1, x2, y2 = map(int, box[:4])
            coordinates = (x1, y1, x2, y2)

            # Crop the detected object from the image
            cropped_image = crop_object(image_path, coordinates)
            # cropped_image.show()  # Display the cropped image
            cropped_image_feature = encode_image(cropped_image)
            indices, values = find_similar_images(cropped_image_feature, dataset_image_features)

            # Print the top matches for the current object
            for idx, value in zip(indices, values):
                match = metadata[idx]
                product = {
                    'name': match['product_name'],
                    'link': match['product_url'],
                    'score': value.item()
                        }
                products.append(product)
    else:
        print("No objects detected.")


    return products

if __name__ == "__main__":
    image_path = 'static/uploads/sample_video_9.png'
    image = Image.open(image_path).convert("RGB")
    hower_image_similarity(image)
