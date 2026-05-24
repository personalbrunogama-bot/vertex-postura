import streamlit as st
import sys
import subprocess

# Instalação forçada de dependências críticas em tempo de execução
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import mediapipe as mp
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Vertex Posture Analytics", layout="centered")

# ... (o restante do código que tínhamos, pode manter a partir daqui) ...
# [Mantenha daqui para baixo o código que já tínhamos, desde a inicialização do mediapipe até ao fim]
