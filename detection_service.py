# detection_service.py - VERSIÓN FINAL TESIS
# Base: V3 (fluida y rápida) +
# Conducta: CentroidTracker + AnalizadorConducta (5 comportamientos) +
# OCR: corrección I/1 con ventana deslizante + fallback rápido (1 variante)

import cv2
import numpy as np
import math
import easyocr
import requests
import time
import threading
import logging
import os
import re
from datetime import datetime
from collections import Counter, deque
from ultralytics import YOLO

import config
import streaming_server

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================
# CONFIGURACIÓN OCR
# ==============================
FRAME_SKIP         = 60    # Procesar cada N frames
MIN_PROB           = 0.75  # Prob. mínima EasyOCR
MAX_HISTORIAL      = 5     # Ventana de consenso
MIN_CONFIRMACIONES = 3     # Lecturas iguales para confirmar placa

# ==============================
# CONFIGURACIÓN DE CONDUCTA
# ==============================
TRACKER_MAX_DIST_PX  = 90   # Distancia máx. centroide para re-asociar persona
TRACKER_MAX_AUSENTE  = 20   # Frames sin ver antes de eliminar track

MERODEO_SEG          = 18   # Segundos en zona de acceso → merodeo
DEAMBULACION_SEG     = 30   # Segundos visibles con mucho recorrido → deambulación
DEAMBULACION_DIST_PX = 250  # Píxeles recorridos mínimos para deambulación
BRUSCO_VEL_PX        = 55   # px/frame para considerar movimiento brusco
BRUSCO_VEL_FRAMES    = 3    # Frames consecutivos con esa velocidad
ENTRADA_OSC_FRAMES   = 8    # Frames de oscilación en zona → forcejeo
ENTRADA_OSC_RADIO_PX = 40   # Radio de oscilación en px

# Zona de acceso como fracción del frame (x1%, y1%, x2%, y2%)
# Ajustar según dónde esté la puerta en tu instalación
ZONA_ACCESO          = (0.20, 0.45, 0.80, 1.00)

COOLDOWN_ALERTA_SEG  = 45   # Segundos mínimos entre alertas del mismo tipo

# ==============================
# MODELOS (se cargan una sola vez)
# ==============================
reader = easyocr.Reader(['es'], gpu=False)  # gpu=True si tienes CUDA
model  = YOLO("yolov8n.pt")

active_processors = {}   # cam_id -> Thread


# ======================================================================
# CORRECCIÓN Y VALIDACIÓN DE PLACA PERUANA
# ======================================================================

# Caracteres visualmente similares según zona de la placa
# Zona LETRAS (pos 1-2): si OCR devuelve dígito, convertir a letra equivalente
_DIG_A_LET = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S',
               '6': 'G', '8': 'B', '9': 'Q'}

# Zona DÍGITOS (pos 3-5): si OCR devuelve letra, convertir a dígito equivalente
_LET_A_DIG = {'O': '0', 'I': '1', 'Z': '2', 'E': '3',
               'A': '4', 'S': '5', 'G': '6', 'T': '7',
               'B': '8', 'Q': '0', 'D': '0'}


def _corregir_6chars(texto6):
    """
    Recibe exactamente 6 caracteres alfanuméricos (A-Z0-9).
    Aplica corrección mínima para placas peruanas y retorna el texto corregido o None.

    Placas peruanas modernas permiten alfanumérico en posiciones 0-2, dígitos en 3-5.
    Solo corregir errores visuales obvios, no forzar conversiones posicionales.

    Ejemplos:
      'A1B234' → mantiene 'A1B234' (válido: A=letra, 1=dígito, B=letra, 234=dígitos)
      'AIB234' → mantiene 'AIB234' (válido: todas letras en 0-2, dígitos en 3-5)
      '1AB123' → mantiene '1AB123' (válido)
      'IAB1I3' → pos 4: 'I'→'1' → 'IAB113' (corrige I por 1 en zona dígitos)
      'ABC4S6' → pos 4: 'S'→'5' → 'ABC456' (corrige S por 5 en zona dígitos)
    """
    c = list(texto6)

    # Posiciones 0-2: alfanumérico - permitir letras y dígitos, no forzar conversiones
    # Solo corregir errores visuales muy obvios si es necesario

    # Posiciones 3-5: zona dígitos - convertir letras claramente similares a dígitos
    for i in range(3, 6):
        if c[i].isalpha():
            c[i] = _LET_A_DIG.get(c[i], c[i])

    resultado = ''.join(c)
    # Validar formato: posiciones 0-2 alfanuméricas, 3-5 dígitos
    if re.match(r'^[A-Z0-9]{3}[0-9]{3}$', resultado):
        return resultado
    return None


def formatear_placa(texto):
    """
    Pipeline completo de validación con 3 pasos:
      1. Limpieza de caracteres no alfanuméricos.
      2. Si tiene exactamente 6 chars → corrección directa.
      3. Si tiene 5-8 chars → ventana deslizante de 6 chars.
         Esto captura cuando el OCR añade/quita 1-2 chars por ruido
         (ej: '1IAB123' → prueba 'IAB123' → corrige → 'IAB-123').
    Retorna 'ABC-123' o None.
    """
    limpio = re.sub(r'[^A-Z0-9]', '', texto.upper().strip())

    # Paso 1: longitud exacta
    if len(limpio) == 6:
        corregido = _corregir_6chars(limpio)
        if corregido:
            return corregido[:3] + '-' + corregido[3:]

    # Paso 2: ventana deslizante para textos de 5 a 8 chars
    if 5 <= len(limpio) <= 8:
        for inicio in range(len(limpio) - 5):
            ventana = limpio[inicio:inicio + 6]
            if len(ventana) < 6:
                continue
            corregido = _corregir_6chars(ventana)
            if corregido:
                return corregido[:3] + '-' + corregido[3:]

    return None


def es_texto_peruano(texto):
    """Filtra la leyenda 'PERU' que aparece en las placas."""
    return texto.strip().upper() in {"PERU", "PERÚ", "P", "PE", "PER"}


# ======================================================================
# PREPROCESAMIENTO DE IMAGEN
# ======================================================================

def _escalar(img, factor=2):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor),
                      interpolation=cv2.INTER_CUBIC)


def preprocesar_variantes(imagen_gris):
    """
    4 variantes para ROI de vehículo (imagen pequeña → merece el costo).
    Cada variante ataca un tipo distinto de degradación de imagen.
    """
    img = _escalar(imagen_gris)
    variantes = []

    # V1 — CLAHE + Bilateral: sombras / bajo contraste
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v1 = clahe.apply(img)
    v1 = cv2.bilateralFilter(v1, 11, 17, 17)
    variantes.append(v1)

    # V2 — Umbral adaptativo: iluminación irregular / soleado directo
    v2 = cv2.GaussianBlur(img, (5, 5), 0)
    v2 = cv2.adaptiveThreshold(v2, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 4)
    variantes.append(v2)

    # V3 — Otsu + cierre morfológico: fondos sucios / marcas
    _, v3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    v3 = cv2.morphologyEx(v3, cv2.MORPH_CLOSE, kernel)
    variantes.append(v3)

    # V4 — Ecualización + sharpen: imágenes borrosas / movidas
    v4 = cv2.equalizeHist(img)
    sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    v4 = cv2.filter2D(v4, -1, sharpen_k)
    variantes.append(v4)

    return variantes


def preprocesar_rapido(imagen_gris):
    """
    1 sola variante para el fallback sobre frame completo.
    CLAHE + bilateral es la más robusta en general.
    Sin upscale en frame completo para mantener fluidez.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img = clahe.apply(imagen_gris)
    return [cv2.bilateralFilter(img, 11, 17, 17)]


# ======================================================================
# OCR MULTI-VARIANTE
# ======================================================================

def _ocr_sobre_imagen(imagen_gris):
    """
    Corre EasyOCR y devuelve (placa, prob, bbox) con mayor probabilidad, o None.
    allowlist limita el vocabulario del modelo → menos errores de reconocimiento.
    """
    resultados = reader.readtext(
        imagen_gris,
        detail=1,
        paragraph=False,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    mejor = None
    for (bbox, text, prob) in resultados:
        if prob < MIN_PROB or es_texto_peruano(text):
            continue
        placa = formatear_placa(text)
        if placa and (mejor is None or prob > mejor[1]):
            mejor = (placa, prob, bbox)
    return mejor


def ocr_buscar_placa(imagen_gris, historial, ultima_placa,
                     variantes_fn=preprocesar_variantes):
    """
    Corre OCR sobre cada variante y aplica confirmación por consenso.
    variantes_fn permite pasar preprocesar_rapido para el fallback.
    Retorna (placa_confirmada | None, historial, bbox).
    """
    mejor_global = None

    for variante in variantes_fn(imagen_gris):
        res = _ocr_sobre_imagen(variante)
        if res and (mejor_global is None or res[1] > mejor_global[1]):
            mejor_global = res

    if mejor_global is None:
        return None, historial, None

    placa_candidata, prob, bbox = mejor_global
    logger.debug(f"  OCR candidata: {placa_candidata} (prob={prob:.2f})")

    historial.append(placa_candidata)
    if len(historial) > MAX_HISTORIAL:
        historial.pop(0)

    conteo = Counter(historial)
    placa_ok, cant = conteo.most_common(1)[0]

    if cant >= MIN_CONFIRMACIONES and placa_ok != ultima_placa:
        return placa_ok, historial, bbox

    return None, historial, None


# ======================================================================
# TRACKER DE PERSONAS
# ======================================================================

class PersonaTrack:
    """Historial de movimiento de una persona identificada por el tracker."""

    def __init__(self, track_id, centroid, bbox, ts):
        self.track_id        = track_id
        self.centroids       = deque(maxlen=200)
        self.bboxes          = deque(maxlen=200)
        self.timestamps      = deque(maxlen=200)
        self.frames_ausente  = 0
        self.alertas_dadas   = set()
        self.vel_alta_streak = 0

        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.timestamps.append(ts)

    def update(self, centroid, bbox, ts):
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.timestamps.append(ts)
        self.frames_ausente = 0

    @property
    def centroid_actual(self):
        return self.centroids[-1]

    @property
    def tiempo_visible(self):
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def velocidad_actual(self):
        if len(self.centroids) < 2:
            return 0.0
        cx1, cy1 = self.centroids[-2]
        cx2, cy2 = self.centroids[-1]
        return math.hypot(cx2 - cx1, cy2 - cy1)

    def distancia_total(self):
        cs = list(self.centroids)
        return sum(
            math.hypot(cs[i][0] - cs[i-1][0], cs[i][1] - cs[i-1][1])
            for i in range(1, len(cs))
        )

    def oscilacion_reciente(self, n_frames, radio_px):
        """True si los últimos n_frames centroides oscilan dentro de radio_px."""
        if len(self.centroids) < n_frames:
            return False
        recientes = list(self.centroids)[-n_frames:]
        cx_med = sum(p[0] for p in recientes) / n_frames
        cy_med = sum(p[1] for p in recientes) / n_frames
        return all(
            math.hypot(p[0] - cx_med, p[1] - cy_med) <= radio_px
            for p in recientes
        )


class CentroidTracker:
    """Asigna IDs persistentes a personas entre frames por distancia de centroide."""

    def __init__(self):
        self.tracks  = {}
        self.next_id = 0

    def update(self, boxes_xyxy, timestamp):
        centroids = [
            (int((x1 + x2) / 2), int((y1 + y2) / 2))
            for x1, y1, x2, y2 in boxes_xyxy
        ]

        for t in self.tracks.values():
            t.frames_ausente += 1

        usados_tracks, usadas_dets = set(), set()

        for di, (cx, cy) in enumerate(centroids):
            mejor_id, mejor_dist = None, TRACKER_MAX_DIST_PX
            for tid, track in self.tracks.items():
                if tid in usados_tracks:
                    continue
                d = math.hypot(cx - track.centroid_actual[0],
                               cy - track.centroid_actual[1])
                if d < mejor_dist:
                    mejor_dist, mejor_id = d, tid

            if mejor_id is not None:
                x1, y1, x2, y2 = boxes_xyxy[di]
                self.tracks[mejor_id].update(
                    (cx, cy), (x1, y1, x2, y2), timestamp)
                usados_tracks.add(mejor_id)
                usadas_dets.add(di)

        for di, (cx, cy) in enumerate(centroids):
            if di not in usadas_dets:
                x1, y1, x2, y2 = boxes_xyxy[di]
                self.tracks[self.next_id] = PersonaTrack(
                    self.next_id, (cx, cy), (x1, y1, x2, y2), timestamp)
                self.next_id += 1

        for tid in [tid for tid, t in self.tracks.items()
                    if t.frames_ausente > TRACKER_MAX_AUSENTE]:
            del self.tracks[tid]

        return {tid: t for tid, t in self.tracks.items()
                if t.frames_ausente == 0}


# ======================================================================
# ANALIZADOR DE CONDUCTA
# ======================================================================

def _en_zona(cx, cy, zona, w, h):
    x1 = int(zona[0] * w)
    y1 = int(zona[1] * h)
    x2 = int(zona[2] * w)
    y2 = int(zona[3] * h)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def analizar_conducta(track: PersonaTrack, frame_shape):
    """
    Evalúa un PersonaTrack y devuelve lista de (tipo_evento, descripcion).
    Cada tipo se emite UNA SOLA VEZ por track (track.alertas_dadas).

    Conductas detectadas:
      MERODEO          — quieto en zona de acceso > umbral de tiempo
      DEAMBULACION     — va y viene mucho tiempo con mucho recorrido
      MOVIMIENTO_BRUSCO— spike de velocidad sostenida (lanzar / golpear)
      ENTRADA_FORZADA  — oscila sin avanzar en zona de acceso (forcejeo)
      APROXIMACION     — viene de fuera y entra rápido a zona de acceso
    """
    alertas = []
    if len(track.centroids) < 4:
        return alertas

    h, w      = frame_shape[:2]
    cx, cy    = track.centroid_actual
    t_visible = track.tiempo_visible
    en_acceso = _en_zona(cx, cy, ZONA_ACCESO, w, h)

    # ── 1. MERODEO ────────────────────────────────────────────────────
    if "MERODEO" not in track.alertas_dadas:
        if en_acceso and t_visible >= MERODEO_SEG:
            alertas.append(("MERODEO",
                             f"Persona merodeando en zona de acceso ({int(t_visible)}s)"))
            track.alertas_dadas.add("MERODEO")

    # ── 2. DEAMBULACIÓN ───────────────────────────────────────────────
    if "DEAMBULACION" not in track.alertas_dadas:
        if t_visible >= DEAMBULACION_SEG:
            dist = track.distancia_total()
            if dist >= DEAMBULACION_DIST_PX:
                alertas.append(("DEAMBULACION",
                                 f"Persona deambulando por el área "
                                 f"({int(t_visible)}s, {int(dist)}px recorridos)"))
                track.alertas_dadas.add("DEAMBULACION")

    # ── 3. MOVIMIENTO BRUSCO (lanzar / golpear) ───────────────────────
    vel = track.velocidad_actual()
    track.vel_alta_streak = (track.vel_alta_streak + 1
                             if vel >= BRUSCO_VEL_PX else 0)

    if "MOVIMIENTO_BRUSCO" not in track.alertas_dadas:
        if track.vel_alta_streak >= BRUSCO_VEL_FRAMES:
            alertas.append(("MOVIMIENTO_BRUSCO",
                             f"Movimiento brusco sostenido (v={int(vel)}px/frame)"))
            track.alertas_dadas.add("MOVIMIENTO_BRUSCO")
            track.vel_alta_streak = 0

    # ── 4. ENTRADA FORZADA (forcejeo en puerta) ───────────────────────
    if "ENTRADA_FORZADA" not in track.alertas_dadas:
        if en_acceso and track.oscilacion_reciente(ENTRADA_OSC_FRAMES,
                                                    ENTRADA_OSC_RADIO_PX):
            alertas.append(("ENTRADA_FORZADA",
                             "Posible intento de entrada forzada "
                             "(oscilación en zona de acceso)"))
            track.alertas_dadas.add("ENTRADA_FORZADA")

    # ── 5. APROXIMACIÓN RÁPIDA A ACCESO ──────────────────────────────
    if "APROXIMACION" not in track.alertas_dadas:
        if en_acceso and len(track.centroids) >= 12:
            cx_old, cy_old = list(track.centroids)[-12]
            venia_de_fuera = not _en_zona(cx_old, cy_old, ZONA_ACCESO, w, h)
            despl = math.hypot(cx - cx_old, cy - cy_old)
            if venia_de_fuera and despl >= 80:
                alertas.append(("APROXIMACION",
                                 f"Aproximación rápida a zona de acceso "
                                 f"({int(despl)}px en 12 frames)"))
                track.alertas_dadas.add("APROXIMACION")

    return alertas


# ======================================================================
# UTILIDADES
# ======================================================================

def guardar_imagen(frame, tipo, camara_id):
    """
    Enviar imagen al backend en lugar de guardarla localmente
    """
    try:
        # Convertir frame a bytes
        import cv2
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            logger.error(f"Error codificando imagen para {tipo}")
            return None

        # Preparar datos para envío
        import requests
        import config

        files = {'imagen': ('imagen.jpg', buffer.tobytes(), 'image/jpeg')}
        data = {
            'camaraId': camara_id,
            'tipo': tipo,
            'descripcion': f'Imagen {tipo} detectada automáticamente'
        }

        # Enviar al backend
        response = requests.post(
            f"{config.BACKEND_URL}/api/imagenes/subir-ia",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {config.SERVICE_TOKEN}"},
            timeout=10
        )

        if response.status_code in [200, 201]:
            result = response.json()
            logger.info(f"✅ Imagen {tipo} subida al backend: {result.get('data', {}).get('url', 'N/A')}")
            return result.get('data', {}).get('url')
        else:
            logger.error(f"❌ Error subiendo imagen {tipo}: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"❌ Error enviando imagen {tipo} al backend: {e}")
        return None


def enviar_a_backend(placa=None, camara_id=None, tipo="PLACA", descripcion=""):
    try:
        payload = {
            "placaDetectada": placa,
            "camaraId":       camara_id,
            "imagenUrl":      None,
            "tipoEvento":     tipo,
            "descripcion":    descripcion
        }
        headers = {
            "Authorization": f"Bearer {config.SERVICE_TOKEN}",
            "Content-Type":  "application/json"
        }
        response = requests.post(
            f"{config.BACKEND_URL}/api/accesos/registrar",
            json=payload,
            headers=headers,
            timeout=10
        )
        if response.status_code in [200, 201]:
            logger.info(f"✅ Backend OK: tipo={tipo} placa={placa}")
        else:
            logger.warning(f"⚠️  Backend {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error conexión backend: {e}")


def dibujar_ui(frame, placa_confirmada, nombre_camara="", mensaje_extra=""):
    h, w, _ = frame.shape
    cv2.putText(frame, "SISTEMA DE MONITOREO", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Estado: ACTIVO", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if placa_confirmada:
        cv2.putText(frame, f"Placa: {placa_confirmada}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if mensaje_extra:
        cv2.putText(frame, mensaje_extra, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ahora = datetime.now().strftime("%d/%m/%Y %I:%M:%S %p")
    cv2.putText(frame, ahora, (w - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# ======================================================================
# PROCESADOR POR CÁMARA
# ======================================================================

def procesar_camara(camara):
    """
    Procesador de cámara con reconexión automática infinita.
    Si la cámara se desconecta, espera y vuelve a intentar indefinidamente.
    """
    cam_id   = camara["id"]
    nombre   = camara.get("nombre", f"Camara-{cam_id}")
    rtsp_url = camara["urlStream"]

    # ============================================================
    # BUCLE INFINITO DE RECONEXIÓN
    # ============================================================
    while True:
        logger.info(f"🔄 Intentando conectar a {nombre}...")

        # ---- Intentar abrir la cámara (3 intentos) ----
        cap = None
        for intento in range(3):
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                logger.info(f"✅ Conectado a: {nombre}")
                break
            cap.release()
            logger.warning(f"⚠️  Intento {intento+1}/3 fallido para {nombre}")
            time.sleep(2)
        else:
            # Si fallaron los 3 intentos, esperar y reintentar desde el inicio
            logger.error(f"🚫 No se pudo conectar a {nombre}, esperando 30s...")
            cap = None
            time.sleep(30)
            continue  # Vuelve al inicio del while True

        # ============================================================
        #  ESTADO LOCAL DE LA CÁMARA (se resetea en cada reconexión)
        # ============================================================
        ultima_placa       = None
        historial_lecturas = []
        tracker            = CentroidTracker()
        cooldowns_alerta   = {}          # tipo_alerta → timestamp
        frame_count        = 0
        mensaje_overlay    = ""
        frames_fallidos    = 0
        MAX_FALLIDOS       = 30

        # ============================================================
        #  BUCLE DE PROCESAMIENTO DE FRAMES
        # ============================================================
        while True:
            ret, frame = cap.read()

            # ---- Manejo de frames fallidos ----
            if not ret or frame is None or frame.size == 0:
                frames_fallidos += 1
                logger.warning(f"⚠️  Frame fallido {frames_fallidos}/{MAX_FALLIDOS} en {nombre}")
                if frames_fallidos >= MAX_FALLIDOS:
                    logger.error(f"🔄 Demasiados fallos ({MAX_FALLIDOS}), reconectando {nombre}...")
                    cap.release()
                    time.sleep(5)
                    break  # Sale del bucle interno → se reinicia desde el while externo
                else:
                    time.sleep(1)
                    continue

            # Si llegamos aquí, el frame es válido
            frames_fallidos = 0
            frame_count += 1
            frame = cv2.resize(frame, (640, 360))

            # ============================================================
            #  PROCESAMIENTO CADA FRAME_SKIP FRAMES
            # ============================================================
            if frame_count % FRAME_SKIP == 0:
                try:
                    results = model(frame, verbose=False)

                    placa_encontrada = False
                    personas_boxes   = []

                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])

                            # -- Vehículo → OCR con 4 variantes --
                            if cls in [2, 7] and not placa_encontrada:
                                roi  = frame[y1:y2, x1:x2]
                                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                                placa_ok, historial_lecturas, _ = ocr_buscar_placa(
                                    gray, historial_lecturas, ultima_placa,
                                    variantes_fn=preprocesar_variantes
                                )

                                if placa_ok:
                                    ultima_placa     = placa_ok
                                    placa_encontrada = True
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    logger.info(f"🚗 PLACA CONFIRMADA [{nombre}]: {ultima_placa}")
                                    frame_ui = frame.copy()
                                    dibujar_ui(frame_ui, ultima_placa, nombre)
                                    guardar_imagen(frame_ui, "placa", cam_id)
                                    threading.Thread(
                                        target=enviar_a_backend,
                                        args=(ultima_placa, cam_id),
                                        daemon=True
                                    ).start()

                            # -- Persona → acumular para tracker --
                            elif cls == 0:
                                personas_boxes.append((x1, y1, x2, y2))

                    # ---- Tracker + análisis de conducta ----
                    ahora_ts = time.time()
                    tracks_activos = tracker.update(personas_boxes, ahora_ts)

                    for tid, track in tracks_activos.items():
                        if track.bboxes:
                            bx1, by1, bx2, by2 = track.bboxes[-1]
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                            cv2.putText(frame, f"P{tid}", (bx1, by1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                        for tipo_alerta, descripcion in analizar_conducta(track, frame.shape):
                            ultimo_ts = cooldowns_alerta.get(tipo_alerta, 0)
                            if ahora_ts - ultimo_ts < COOLDOWN_ALERTA_SEG:
                                continue
                            cooldowns_alerta[tipo_alerta] = ahora_ts
                            mensaje_overlay = tipo_alerta.replace("_", " ")
                            logger.warning(f"⚠️  [{nombre}] {descripcion}")
                            frame_alerta = frame.copy()
                            dibujar_ui(frame_alerta, ultima_placa, nombre, mensaje_overlay)
                            guardar_imagen(frame_alerta, tipo_alerta.lower(), cam_id)
                            threading.Thread(
                                target=enviar_a_backend,
                                args=(None, cam_id, "SOSPECHOSA", descripcion),
                                daemon=True
                            ).start()

                    # ---- Fallback OCR: 1 variante sobre frame completo ----
                    if not placa_encontrada:
                        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        placa_ok, historial_lecturas, _ = ocr_buscar_placa(
                            gray_full, historial_lecturas, ultima_placa,
                            variantes_fn=preprocesar_rapido
                        )
                        if placa_ok:
                            ultima_placa = placa_ok
                            logger.info(f"🚗 PLACA fallback [{nombre}]: {ultima_placa}")
                            frame_ui = frame.copy()
                            dibujar_ui(frame_ui, ultima_placa, nombre)
                            guardar_imagen(frame_ui, "placa", cam_id)
                            threading.Thread(
                                target=enviar_a_backend,
                                args=(ultima_placa, cam_id),
                                daemon=True
                            ).start()

                except Exception as e:
                    logger.error(f"❌ Error procesando frame de {nombre}: {e}")

            # ---- UI y streaming ----
            dibujar_ui(frame, ultima_placa, nombre, mensaje_overlay)
            streaming_server.update_camera_frame(cam_id, frame)

            # ---- Salir del bucle (por si se presiona 'q' en otro proceso) ----
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                streaming_server.remove_camera(cam_id)
                logger.info(f"🛑 Procesamiento detenido por usuario en {nombre}")
                return

        # ---- Si salimos del bucle interno por reconexión, liberar y esperar ----
        cap.release()
        logger.info(f"🔄 Reconectando {nombre} en 5 segundos...")
        time.sleep(5)
        # El bucle while True externo continúa automáticamente
# ======================================================================
# MONITOREO DINÁMICO DE CÁMARAS
# ======================================================================

def monitorear_camaras():
    while True:
        try:
            # 1. Limpiar threads muertos
            dead_threads = []
            for cam_id, thread in active_processors.items():
                if not thread.is_alive():
                    dead_threads.append(cam_id)
                    logger.warning(f"🔄 Thread muerto detectado para cámara {cam_id}, será reiniciado")
            
            for cam_id in dead_threads:
                del active_processors[cam_id]

            # 2. Consultar cámaras activas
            resp = requests.get(
                f"{config.BACKEND_URL}/api/camaras/activas",
                headers={"Authorization": f"Bearer {config.SERVICE_TOKEN}"},
                timeout=10
            )
            if resp.status_code == 200:
                camaras = resp.json().get("data", [])
                for cam in camaras:
                    cam_id = cam["id"]
                    if cam.get("activa") and cam_id not in active_processors:
                        t = threading.Thread(
                            target=procesar_camara,
                            args=(cam,),
                            daemon=True
                        )
                        t.start()
                        active_processors[cam_id] = t
                        logger.info(f"📷 Cámara iniciada: {cam.get('nombre')}")
            else:
                logger.warning(f"⚠️  /api/camaras/activas → {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Error consultando cámaras: {e}")
        time.sleep(25)

# ======================================================================
# ENTRADA PRINCIPAL
# ======================================================================

if __name__ == "__main__":
    logger.info("🚀 DepaManager LPR Multi-Cámara iniciado")
    
    # Iniciar servidor de streaming en un thread separado
    threading.Thread(target=streaming_server.app.run, kwargs={'host': '0.0.0.0', 'port': 5001, 'threaded': True}, daemon=True).start()
    logger.info("📡 Servidor de streaming MJPEG iniciado en puerto 5001")
    
    # Iniciar monitoreo de cámaras
    threading.Thread(target=monitorear_camaras, daemon=True).start()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("🛑 Servicio detenido")