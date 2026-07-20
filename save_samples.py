import os
from PIL import Image
from medmnist import OCTMNIST

# 1. Crear una carpeta para las muestras si no existe
output_dir = "retina_samples"
os.makedirs(output_dir, exist_ok=True)

# 2. Descargar el dataset original (SIN transformaciones de PyTorch para conservar el formato imagen)
print("Descargando set de prueba de OCTMNIST...")
raw_test_dataset = OCTMNIST(split="test", download=True)

# Mapeo interno de clases para nombrar los archivos
classes_map = {
    0: "CNV",
    1: "DME",
    2: "DRUSEN",
    3: "NORMAL"
}

print(f"Guardando imágenes de muestra en la carpeta '{output_dir}'...")

# 3. Guardar las primeras 5 imágenes del set de prueba
for idx in range(5):
    # MedMNIST sin transformaciones devuelve una tupla: (Imagen_PIL, Label_Numpy)
    pil_img, label = raw_test_dataset[idx]
    class_name = classes_map[int(label[0])]
    
    # Definir el nombre del archivo
    filename = f"sample_{idx}_{class_name}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Guardar en el disco
    pil_img.save(filepath)
    print(f"-> Guardada: {filepath}")