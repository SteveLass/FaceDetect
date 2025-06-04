import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Configuration de la page
st.set_page_config(page_title="Détection de visages enrichie", layout="wide")
st.title("📸 Bienvenue dans l'application de détection de visages")

st.markdown("""
Détectez automatiquement les **visages** dans une image ou via votre **webcam** grâce à l’algorithme Viola-Jones.

**Fonctionnalités :**
- 📁 Téléversement d'image ou capture 🎥 webcam
- 🎨 Choix de la couleur du cadre
- ⚙️ Ajustement de la précision
- 💾 Téléchargement du résultat
""")

# Classifieur pour les visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Utilitaire : hex ➝ BGR
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]

# Fonction de détection de visages uniquement
def detect_faces_only(image_cv, scaleFactor, minNeighbors, color_bgr):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:

        cv2.rectangle(image_cv, (x, y), (x + w, y + h), color_bgr, 2)

        
    return image_cv, len(faces)

# === SIDEBAR : Paramètres ===
st.sidebar.header("⚙️ Paramètres")
source = st.sidebar.radio("📷 Source", ["Téléverser une image", "Utiliser la webcam"])
scaleFactor = st.sidebar.slider("🔍 scaleFactor", 1.01, 1.5, 1.1, 0.01)
minNeighbors = st.sidebar.slider("👥 minNeighbors", 1, 10, 5)

color_face = hex_to_bgr(st.sidebar.color_picker("🎨 Couleur du rectangle", "#00FF00"))

# === MODE : Téléversement ===
if source == "Téléverser une image":
    file = st.file_uploader("📁 Téléversez une image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="🖼️ Image originale", use_column_width=True)
        with col2:
            if st.button("🚀 Lancer la détection"):
                result, faces_count = detect_faces_only(image_cv.copy(), scaleFactor, minNeighbors, color_face)
                st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="🧠 Résultat", use_column_width=True)
                st.success(f"✅ {faces_count} visage(s) détecté(s)")

                # Téléchargement de l'image résultante

                result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                result_pil.save(buffer.name)

                with open(buffer.name, "rb") as f:
                    st.download_button("📥 Télécharger l'image détectée", f, file_name="visages_detectes.jpg", mime="image/jpeg")

# === MODE : Webcam ===
else:
    st.info("Cochez pour démarrer la webcam ⬇️")
    start_cam = st.checkbox("🎥 Démarrer la webcam")
    cam_placeholder = st.empty()

    if start_cam:
        cap = cv2.VideoCapture(0)
        st.warning("Appuyez sur **⛔ Stop** pour arrêter la caméra.")
        stop = st.button("⛔ Stop")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Problème de lecture de la webcam.")
                break

            result, _ = detect_faces_only(frame.copy(), scaleFactor, minNeighbors, color_face)
            cam_placeholder.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        cap.release()
