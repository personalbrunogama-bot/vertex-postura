import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

# 1. Configuração estrita da página (deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Vertex Posture Analytics",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Estilização CSS para interface Mobile-Friendly e Profissional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e293b; font-family: 'Helvetica Neue', Arial, sans-serif; text-align: center; font-size: 24px; }
    .metric-box {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        border-left: 5px solid #3b82f6;
    }
    .metric-title { font-weight: bold; color: #334155; margin-bottom: 4px; font-size: 16px; }
    .metric-value { font-size: 15px; color: #475569; margin: 2px 0; }
    </style>
""", unsafe_allow_html=True)

# 3. Inicialização estável do MediaPipe Pose
@st.cache_resource
def inicializar_mediapipe():
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    return mp_pose, pose_detector

try:
    mp_pose, pose = inicializar_mediapipe()
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"Erro ao inicializar engine biomecânica: {e}")

def calcular_angulo_horizontal(p1, p2):
    """Calcula o ângulo estrito em relação à linha do horizonte."""
    dy = p2.y - p1.y
    dx = p2.x - p1.x
    angulo_rad = math.atan2(dy, dx)
    return math.degrees(angulo_rad)

# Interface do Usuário
st.title("🧬 Vertex Posture Analytics")
st.markdown("<p style='text-align: center; color: #64748b;'>Análise Biomecânica Digital e Postural</p>", unsafe_allow_html=True)

# Seleção de Origem da Imagem otimizada para celular
opcao_captura = st.radio("Selecione a origem da foto:", ("Usar Câmera do Celular", "Escolher da Galeria / Upload"))

imagem_pil = None

if opcao_captura == "Usar Câmera do Celular":
    arquivo_camera = st.camera_input("Posicione o cliente em vista frontal/ortostática")
    if arquivo_camera is not None:
        imagem_pil = Image.open(arquivo_camera)
else:
    arquivo_upload = st.file_uploader("Selecione o arquivo de imagem", type=["jpg", "jpeg", "png"])
    if arquivo_upload is not None:
        imagem_pil = Image.open(arquivo_upload)

# Processamento Analítico
if imagem_pil is not None:
    # Conversão robusta para formato OpenCV (Numpy array RGB)
    imagem_np = np.array(imagem_pil)
    
    # Tratamento de imagens com canal alpha (RGBA) vindas de alguns celulares
    if len(imagem_np.shape) == 3 and imagem_np.shape[2] == 4:
        imagem_np = cv2.cvtColor(imagem_np, cv2.COLOR_RGBA2RGB)
    elif len(imagem_np.shape) == 2:
        imagem_np = cv2.cvtColor(imagem_np, cv2.COLOR_GRAY2RGB)
        
    imagem_output = imagem_np.copy()
    h, w, _ = imagem_output.shape
    
    with st.spinner("Mapeando pontos anatômicos (Acrômios e Pélvis)..."):
        resultados = pose.process(imagem_output)
        
    if resultados.pose_landmarks:
        st.success("✅ Mapeamento concluído!")
        
        # Extração das coordenadas normalizadas
        pontos = resultados.pose_landmarks.landmark
        
        ombro_esq = pontos[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        ombro_dir = pontos[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        quadril_esq = pontos[mp_pose.PoseLandmark.LEFT_HIP.value]
        quadril_dir = pontos[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Cálculo exato dos ângulos absolutos
        angulo_ombros = abs(calcular_angulo_horizontal(ombro_esq, ombro_dir))
        angulo_quadril = abs(calcular_angulo_horizontal(quadril_esq, quadril_dir))
        
        # Ajuste de rotação geométrica para exibição de pequenos ângulos corporais
        if angulo_ombros > 90: angulo_ombros = abs(angulo_ombros - 180)
        if angulo_quadril > 90: angulo_quadril = abs(angulo_quadril - 180)
        
        # Determinação clínica precisa do lado deprimido (Maior Y = Mais baixo na imagem)
        # Nota: LEFT/RIGHT no MediaPipe refere-se ao lado anatômico do próprio cliente
        lado_baixo_ombro = "Esquerdo (Anatômico)" if ombro_esq.y > ombro_dir.y else "Direito (Anatômico)"
        lado_baixo_quadril = "Esquerdo (Anatômico)" if quadril_esq.y > quadril_dir.y else "Direito (Anatômico)"
        
        # Desenho das conexões esqueléticas na imagem
        mp_drawing.draw_landmarks(
            imagem_output, 
            resultados.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3), # Pontos brancos
            mp_drawing.DrawingSpec(color=(246, 130, 59), thickness=2, circle_radius=2)  # Linhas laranjas
        )
        
        # Traçar linhas de horizonte digital guias (Azul para ombros, Verde para quadril)
        cv2.line(imagem_output, (0, int(ombro_esq.y * h)), (w, int(ombro_esq.y * h)), (59, 130, 246), 2)
        cv2.line(imagem_output, (0, int(quadril_esq.y * h)), (w, int(quadril_esq.y * h)), (16, 185, 129), 2)
        
        # Exibição dos resultados adaptada para telas de celulares (Disposição vertical limpa)
        st.image(imagem_output, caption="Rastreamento Biomecânico", use_container_width=True)
        
        st.markdown("### 📊 Relatório de Assimetria")
        
        # Classificação clínica simplificada
        status_ombro = "🟢 Alinhado" if angulo_ombros < 1.5 else "🟡 Desvio Sutil" if angulo_ombros < 3.0 else "🔴 Desvio Significativo"
        status_quadril = "🟢 Alinhado" if angulo_quadril < 1.5 else "🟡 Desvio Sutil" if angulo_quadril < 3.0 else "🔴 Desvio Significativo"
        
        # Bloco de Métricas - Ombros
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Linha dos Ombros (Acrômios)</div>
            <div class="metric-value"><b>Desnível Angular:</b> {angulo_ombros:.2f}°</div>
            <div class="metric-value"><b>Classificação:</b> {status_ombro}</div>
            <div class="metric-value"><small>Hemicorpo deprimido: {lado_baixo_ombro}</small></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bloco de Métricas - Quadril
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: #10b981;">
            <div class="metric-title">Linha da Pélvis (Báscula Pélvica)</div>
            <div class="metric-value"><b>Inclinação Axial:</b> {angulo_quadril:.2f}°</div>
            <div class="metric-value"><b>Classificação:</b> {status_quadril}</div>
            <div class="metric-value"><small>Hemicorpo deprimido: {lado_baixo_quadril}</small></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Campo de anotações práticas para o avaliador
        st.markdown("### 📝 Conduta e Evolução")
        notas = st.text_area("Observações clínicas e direcionamento de exercício:", 
                             placeholder="Ex: Foco em ativação de glúteo médio do lado deprimido, acompanhar evolução...")
        
    else:
        st.error("⚠️ Não foi possível detectar a postura. Certifique-se de que o corpo inteiro do cliente (dos ombros ao quadril) esteja visível, com roupas adequadas e boa iluminação.")
