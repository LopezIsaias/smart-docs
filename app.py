import streamlit as st
import pandas as pd
import time
from io import BytesIO
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import json
import os
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

# --- CARGAR VARIABLES DE ENTORNO ---
# En local busca .env; en AWS leerá el archivo .env creado por tu script de User Data
load_dotenv()

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================

# Obtenemos la clave de manera segura desde el entorno
API_KEY_GOOGLE = os.getenv("GOOGLE_API_KEY")

# NOTA: En AWS (Linux), Tesseract se detecta automáticamente en el PATH.
# No es necesario configurar pytesseract.tesseract_cmd.

# PROMPT MAESTRO (Tu configuración personalizada)
SYSTEM_PROMPT = """
Eres un asistente de IA especializado en el procesamiento y extracción de datos de documentos administrativos y académicos (Oficios, Cartas, Resoluciones, Informes, Memorandos, etc.).

Tarea: Tu objetivo es analizar el texto proporcionado, identificar la naturaleza del documento y extraer metadatos clave para estructurarlos en un formato JSON válido. No debes añadir ningún texto, saludo o explicación fuera del bloque JSON.

Reglas de Extracción y Limpieza:

Tipo de Documento: Identifica la clase de documento (ej. Resolución, Oficio, Carta, Informe, Memorando, etc.).

Número de Documento:

Extrae solo la parte numérica significativa.

Elimina ceros a la izquierda y cualquier prefijo o letra (ej. "D000496" se convierte en "496"; "001" se convierte en "1").

Institución y Facultad: Extrae los nombres completos.

Ciudad:

Extrae la ciudad si se menciona explícitamente (generalmente en la línea de fecha).

Regla estricta: Si la ciudad no figura en el texto, devuelve una cadena vacía "".

Fecha:

Estandariza al formato DD/MM/AAAA.

Emisores y Receptores:

Conserva los grados académicos y títulos tal como aparecen (ej. "Ing. M.Sc.", "Dr.", "Lic.").

Separa múltiples nombres con comas.

Lógica para "Resoluciones":

Emisor: Son las autoridades que firman al final del documento (generalmente después de la frase "Regístrese, comuníquese y archívese").

Receptor: Son las personas mencionadas en la sección resolutiva ("SE RESUELVE") sobre las cuales recae la acción (ej. estudiantes sancionados, tesistas aprobados, etc.).

Lógica para otros documentos (Oficios, Cartas, etc.):

Emisor: Quien firma al pie o figura en el encabezado como remitente.

Receptor: A quien va dirigido el documento ("Señor...", "Al...").

Referencia:

Extrae el texto de la referencia.

Si el texto contiene una fecha (ej. "15Agosto2024"), conviértela a formato numérico entre paréntesis: (15/08/2024).

Si no hay referencia explícita, devuelve una cadena vacía "".

Resumen Ejecutivo: Redacta una síntesis precisa del propósito del documento.

Formato de Texto: Asegura que exista un espacio después de cada punto (.), excepto en siglas o grados académicos compactos.

Formato de Salida (JSON):

Debes devolver únicamente un objeto JSON con la siguiente estructura:

{
  "tipo_documento": "String",
  "numero_documento": "String",
  "institucion": "String",
  "facultad": "String",
  "ciudad": "String",
  "fecha": "String",
  "nombre_emisor": "String",
  "nombre_receptor": "String",
  "referencia": "String",
  "resumen_ejecutivo": "String"
}
"""

# Configurar la API de Google
if not API_KEY_GOOGLE:
    # Esto aparecerá si el script de User Data falló al crear el .env
    st.error("❌ ERROR DE CONFIGURACIÓN: No se encontró la API Key. Verifica que el archivo .env exista en el servidor.")
    st.stop()
else:
    genai.configure(api_key=API_KEY_GOOGLE)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SmartDocs AI - Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; color: #2C3E50; }
    h1, h2, h3 { color: #2C3E50 !important; font-weight: 700; }
    div.stButton > button[kind="primary"] {
        background-color: #27AE60; color: white; border: none; border-radius: 6px;
        padding: 0.5rem 1rem; font-weight: bold; transition: all 0.3s ease;
    }
    div.stButton > button[kind="primary"]:hover { background-color: #219150; }
    .stFileUploader { background-color: #F8F9FA; border: 1px dashed #6C5CE7; border-radius: 10px; padding: 20px; }
    div[data-testid="column"] { display: flex; align-items: center; }
    </style>
    """, unsafe_allow_html=True)

# --- ESTADO DE LA SESIÓN ---
if 'datos_procesados' not in st.session_state:
    st.session_state.datos_procesados = []

# ==============================================================================
# 2. FUNCIONES DE LÓGICA
# ==============================================================================

def extraer_texto_hibrido(archivo_bytes):
    """
    Procesa el PDF directamente desde los bytes en memoria RAM.
    """
    texto_final = []
    
    try:
        # Abrir PDF desde el stream de memoria
        documento = fitz.open(stream=archivo_bytes, filetype="pdf")
        
        for pagina in documento:
            elementos_pagina = []
            
            # 1. Texto seleccionable
            bloques_texto = pagina.get_text("blocks")
            for bloque in bloques_texto:
                elementos_pagina.append({'tipo': 'texto', 'bbox': bloque[:4], 'contenido': bloque[4]})

            # 2. Imágenes (OCR)
            try:
                imagenes = pagina.get_images(full=True)
                for img_info in imagenes:
                    xref = img_info[0]
                    try:
                        bbox = pagina.get_image_bbox(img_info)
                        elementos_pagina.append({'tipo': 'imagen', 'bbox': bbox, 'xref': xref})
                    except:
                        pass
            except:
                pass

            # Ordenar elementos visualmente
            elementos_pagina.sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))

            for elemento in elementos_pagina:
                if elemento['tipo'] == 'texto':
                    texto_final.append(elemento['contenido'])
                elif elemento['tipo'] == 'imagen':
                    try:
                        img_bytes = documento.extract_image(elemento['xref'])["image"]
                        imagen = Image.open(io.BytesIO(img_bytes))
                        # Tesseract en Linux funciona directo si está instalado con apt-get
                        texto_ocr = pytesseract.image_to_string(imagen, lang='spa')
                        if texto_ocr.strip():
                            texto_final.append(texto_ocr)
                    except Exception:
                        pass # Ignorar fallos puntuales de imagen

        documento.close()
        
        # Limpieza de espacios
        contenido_completo = "\n".join(texto_final)
        contenido_limpio = re.sub(r'\n\s*\n', '\n', contenido_completo)
        contenido_limpio = re.sub(r'[ \t]+', ' ', contenido_limpio)
        
        return contenido_limpio.strip()

    except Exception as e:
        return f"Error al leer PDF: {str(e)}"

def consultar_gemini(texto_pdf):
    """Envía el texto a Gemini y parsea el JSON"""
    try:
        # Intentamos usar el modelo 2.5-flash solicitado
        try:
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            prompt_completo = f"{SYSTEM_PROMPT}\n\nDOCUMENTO A ANALIZAR:\n{texto_pdf}"
            response = model.generate_content(prompt_completo)
        except Exception:
            # Fallback robusto al modelo estándar si el experimental falla en la región AWS
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            prompt_completo = f"{SYSTEM_PROMPT}\n\nDOCUMENTO A ANALIZAR:\n{texto_pdf}"
            response = model.generate_content(prompt_completo)
        
        # Limpieza de respuesta Markdown (común en respuestas JSON)
        texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
        datos_json = json.loads(texto_limpio)
        
        return datos_json

    except Exception as e:
        return {"Error": f"Fallo en IA: {str(e)}"}

# ==============================================================================
# 3. INTERFAZ GRÁFICA
# ==============================================================================

col_logo, col_titulo, col_accion = st.columns([0.5, 4, 1.5])
with col_titulo:
    st.title("SmartDocs AI")
    st.markdown("<span style='color: #6C5CE7; font-weight: bold;'>⚡ Extracción Inteligente (AWS Cloud)</span>", unsafe_allow_html=True)

# Lógica de Excel
df_final = pd.DataFrame(st.session_state.datos_procesados)
excel_buffer = BytesIO()
boton_deshabilitado = True

if not df_final.empty:
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_final.to_excel(writer, index=False, sheet_name='Datos_Extraidos')
    datos_excel = excel_buffer.getvalue()
    boton_deshabilitado = False
else:
    datos_excel = b""

with col_accion:
    st.write("") 
    st.download_button(
        label="📥 Descargar Excel",
        data=datos_excel,
        file_name="reporte_ia.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        disabled=boton_deshabilitado,
        use_container_width=True
    )

st.markdown("---")
c_sidebar, c_main = st.columns([1, 2.5], gap="large")

# Panel Izquierdo
with c_sidebar:
    st.subheader("📥 Documentos")
    archivos = st.file_uploader("Arrastra archivos aquí", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if archivos:
        st.success(f"📂 {len(archivos)} archivos cargados")
        barra_progreso = st.progress(0)
        
        # Obtenemos nombres ya procesados para evitar duplicados
        nombres_memoria = [d.get('Nombre Archivo', 'Desconocido') for d in st.session_state.datos_procesados]
        nuevos_procesados = 0
        
        for i, archivo in enumerate(archivos):
            if archivo.name not in nombres_memoria:
                # 1. Lectura (Desde RAM)
                with st.spinner(f"👁️ Leyendo {archivo.name}..."):
                    bytes_pdf = archivo.read()
                    texto_extraido = extraer_texto_hibrido(bytes_pdf)
                
                # 2. IA
                if "Error" not in texto_extraido:
                    with st.spinner(f"🧠 Analizando..."):
                        datos_ia = consultar_gemini(texto_extraido)
                        
                        if "Error" in datos_ia:
                            st.error(f"Error en {archivo.name}: {datos_ia['Error']}")
                        else:
                            # Añadir nombre y guardar
                            ordenado = {'Nombre Archivo': archivo.name}
                            ordenado.update(datos_ia)
                            st.session_state.datos_procesados.append(ordenado)
                            nuevos_procesados += 1
                else:
                    st.error(f"Fallo al leer {archivo.name}: {texto_extraido}")

            barra_progreso.progress((i + 1) / len(archivos))
            
        if nuevos_procesados > 0:
            st.rerun()

    if st.session_state.datos_procesados:
        st.write("---")
        st.markdown("**Historial:**")
        for item in st.session_state.datos_procesados:
            st.markdown(f"✅ <small>{item.get('Nombre Archivo')}</small>", unsafe_allow_html=True)

# Panel Central
with c_main:
    st.subheader("📊 Datos Extraídos")
    
    if not df_final.empty:
        st.info("Puedes editar celdas o borrar filas con la tecla 'Supr'.")
        
        df_editado = st.data_editor(
            df_final,
            num_rows="dynamic",
            use_container_width=True,
            height=600,
            key="data_editor",
            column_config={
                "fecha": st.column_config.TextColumn("Fecha Emisión"),
                # Puedes agregar más configuraciones de columna aquí si deseas
            }
        )
        
        # Sincronización al borrar
        if len(df_editado) < len(st.session_state.datos_procesados):
            st.toast("Registro eliminado", icon="🗑️")
            st.session_state.datos_procesados = df_editado.to_dict('records')
            time.sleep(1)
            st.rerun()
            
    else:
        st.markdown("""
            <div style='text-align: center; color: #95a5a6; padding: 50px; background-color: #F8F9FA; border-radius: 10px;'>
                <h2>👋 Listo para empezar</h2>
                <p>Sube tus documentos oficiales.</p>
            </div>
        """, unsafe_allow_html=True)