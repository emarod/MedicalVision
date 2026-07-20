import streamlit as pd_st
import requests
from PIL import Image
import io

# Configuración estética de la página estilo clínica digital
pd_st.set_page_config(
    page_title="App Tecnología Oftalmológica",
    page_icon="👁️",
    layout="centered"
)

pd_st.title("Sistema de Tamizaje Oftalmológico Digital")
pd_st.markdown("""
Bienvenido al portal médico auxiliar para la evaluación de tomografías de coherencia óptica (OCT) de retina. 
Suba una imagen de la retina del paciente para obtener un pre-diagnóstico asistido por Inteligencia Artificial.
""")

# URL de la API de FastAPI (cuando corra local en Docker, usaremos la red interna o localhost)
API_URL = "http://localhost:8000/predict"

uploaded_file = pd_st.file_uploader("Seleccione una imagen de retina (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen en la UI
    image = Image.open(uploaded_file)
    pd_st.image(image, caption="Imagen cargada por el médico", use_container_width=False)
    
    # Botón para detonar el diagnóstico
    if pd_st.button("Generar Diagnóstico Asistido"):
        with pd_st.spinner("Ejecutando pipeline de inferencia en el servidor..."):
            try:
                # Convertir la imagen a bytes para enviarla vía multipart/form-data
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else "PNG")
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                
                # Consumir la API
                response = requests.post(API_URL, files=files, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Mostrar resultados principales
                    pd_st.success(f"**Predicción Principal:** {result['prediction']}")
                    pd_st.metric(label="Nivel de Confianza", value=f"{result['confidence']}%")
                    
                    # Desglose de probabilidades en barras
                    pd_st.subheader("Desglose Probabilístico por Patología")
                    for condition, percentage in result['all_probabilities'].items():
                        pd_st.write(f"**{condition}**")
                        pd_st.progress(percentage / 100.0)
                        pd_st.caption(f"Probabilidad estimada: {percentage}%")
                        
                else:
                    pd_st.error(f"Error en el servidor de inferencia: {response.text}")
            except Exception as e:
                pd_st.error(f"No se pudo establecer conexión con la API de inferencia: {e}")