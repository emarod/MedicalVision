import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from model import Net

app = FastAPI(
    title="API Vision Médica",
    description="Servicio REST para la clasificación automática de tomografías de coherencia óptica (OCT) de retina.",
    version="1.0.0"
)

CLASSES_MAP = {
    0: "Choroidal Neovascularization (CNV)",
    1: "Diabetic Macular Edema (DME)",
    2: "Drusen",
    3: "Normal"
}

device = torch.device('cpu')
model = Net(in_channels=1, num_classes=4)

try:
    weights = torch.load("octmnist_cnn_weights.pth", map_location=device)
    model.load_state_dict(weights)
    model.eval()
    print("Modelo cargado en CPU.")
except Exception as e:
    print(f"Error al cargar los pesos del modelo: {e}")

transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Endpoint de prueba
@app.get("/", tags=["General"])
def health_check():
    return {"status": "healthy", "model_loaded": True}

# Endpoint Inferencia
@app.post("/predict", tags=["Machine Learning"])
async def predict_retina(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo subido debe ser una imagen.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))        
        # Aplicar el pipeline de preprocesamiento e insertar dimensión de Batch [1, 1, 28, 28]
        tensor_image = transform_pipeline(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_image)
            probabilities = torch.softmax(outputs, dim=-1).squeeze()
            
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()
        
        return {
            "prediction": CLASSES_MAP[predicted_class_idx],
            "class_index": predicted_class_idx,
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                CLASSES_MAP[i]: round(prob.item() * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la inferencia: {str(e)}")