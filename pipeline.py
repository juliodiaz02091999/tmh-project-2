# ============================================================
# PIPELINE GENERAL PARA:
# - Segmentación del ojo con el modelo ONNX oficial de Open-IRIS
# - Detección del reflejo especular del menisco lagrimal
# - Estimación automática del TMH
# - Procesamiento de una imagen o una carpeta completa
#
# NO REQUIERE instalar open-iris.
#
# Salidas:
# - máscaras de segmentación
# - ROI del menisco
# - candidatos de reflejo
# - reflejo seleccionado
# - trazado de bordes del menisco
# - TMH en píxeles y, opcionalmente, en mm aproximados
# ============================================================

from pathlib import Path
import glob
import json
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort

from huggingface_hub import hf_hub_download
try:
    from IPython.display import display
except ImportError:
    def display(obj):  # noqa: F811
        pass


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta de una sola imagen:
INPUT_PATH = "/kaggle/input/datasets/juliochdiaz0209/imagen1/tmh10.PNG"

# Para procesar una carpeta completa, reemplazar por:
# INPUT_PATH = "/kaggle/input/datasets/juliochdiaz0209/imagen1"

OUTPUT_DIR = "/kaggle/working/tmh_general_onnx_output"

# El modelo oficial se descargará automáticamente.
# Si ya descargaste/subiste manualmente el archivo ONNX, coloca aquí su ruta.
LOCAL_MODEL_PATH = None
# Ejemplo:
# LOCAL_MODEL_PATH = "/kaggle/input/modelos/iris_semseg_upp_scse_mobilenetv2.onnx"

MODEL_REPO = "Worldcoin/iris-semantic-segmentation"
MODEL_FILENAME = "iris_semseg_upp_scse_mobilenetv2.onnx"

# El modelo trabaja internamente con esta resolución.
MODEL_WIDTH = 640
MODEL_HEIGHT = 480

# Umbrales de máscaras semánticas.
SEG_THRESHOLD = {
    "eyeball": 0.50,
    "iris": 0.50,
    "pupil": 0.50,
    "eyelashes": 0.50,
}

# Conversión opcional a milímetros.
# Opción recomendada: colocar MM_PER_PIXEL si calibras tu sistema físicamente.
MM_PER_PIXEL = None

# Solo si no existe calibración física:
# conversión aproximada empleando un diámetro horizontal de iris/córnea.
REFERENCE_IRIS_DIAMETER_MM = 11.5

# Rango físico admisible de TMH para restringir búsqueda del borde superior.
# Puede modificarse luego de revisar tu base completa.
TMH_MIN_MM = 0.05
TMH_MAX_MM = 0.80

# Extensiones admitidas cuando INPUT_PATH es una carpeta.
IMAGE_EXTENSIONS = [
    "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"
]

SHOW_PLOTS = True
SAVE_INTERMEDIATE_IMAGES = True


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def odd(value, minimum=3):
    """Convierte un número a entero impar, respetando un mínimo."""
    number = max(minimum, int(round(value)))
    return number if number % 2 == 1 else number + 1


def json_serializable(value):
    """Convierte tipos NumPy a tipos serializables en JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def largest_component(mask):
    """Conserva únicamente el componente conectado de mayor área."""
    binary = (mask > 0).astype(np.uint8)

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if number <= 1:
        return binary

    largest_id = 1 + int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    return (labels == largest_id).astype(np.uint8)


def fill_holes(mask):
    """Rellena huecos internos en una máscara binaria."""
    binary = (mask > 0).astype(np.uint8) * 255

    flood = binary.copy()
    h, w = binary.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(
        flood,
        flood_mask,
        seedPoint=(0, 0),
        newVal=255
    )

    holes = cv2.bitwise_not(flood)

    filled = cv2.bitwise_or(binary, holes)

    return (filled > 0).astype(np.uint8)


def smooth_1d(values, kernel_size):
    """Suavizado gaussiano unidimensional."""
    kernel_size = odd(kernel_size, minimum=5)

    array = values.astype(np.float32).reshape(1, -1)

    smoothed = cv2.GaussianBlur(
        array,
        (kernel_size, 1),
        0
    )

    return smoothed.ravel()


def enhance_gray(gray):
    """Realza contraste local preservando bordes."""
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.bilateralFilter(
        enhanced,
        d=5,
        sigmaColor=20,
        sigmaSpace=20
    )

    return enhanced


def resolve_image_paths(input_path):
    """Resuelve una ruta individual, una carpeta o un patrón glob."""
    path = Path(input_path)

    if path.is_file():
        return [path]

    if path.is_dir():
        images = []

        for extension in IMAGE_EXTENSIONS:
            images.extend(path.glob(extension))

        return sorted(images)

    return sorted([Path(item) for item in glob.glob(input_path)])


# ============================================================
# MODELO ONNX OFICIAL DE OPEN-IRIS
# ============================================================

class OfficialOpenIrisONNXSegmenter:
    """
    Ejecución directa del modelo ONNX oficial de Open-IRIS.

    Clases:
        0 -> eyeball
        1 -> iris
        2 -> pupil
        3 -> eyelashes
    """

    CLASS_MAPPING = {
        "eyeball": 0,
        "iris": 1,
        "pupil": 2,
        "eyelashes": 3,
    }

    def __init__(self, local_model_path=None):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        if local_model_path is not None:
            model_path = Path(local_model_path)

            if not model_path.exists():
                raise FileNotFoundError(
                    f"No existe el modelo ONNX local: {model_path}"
                )

            self.model_path = str(model_path)

        else:
            print("Descargando/cargando modelo ONNX oficial de Open-IRIS...")

            try:
                self.model_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILENAME
                )

            except Exception as error:
                raise RuntimeError(
                    "No se pudo descargar el modelo ONNX. "
                    "Activa Internet en Kaggle o asigna una ruta en "
                    "LOCAL_MODEL_PATH."
                ) from error

        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
        )

        model_input = self.session.get_inputs()[0]

        self.input_name = model_input.name
        self.input_dtype = (
            np.float16
            if "float16" in model_input.type.lower()
            else np.float32
        )

        print(f"Modelo cargado: {self.model_path}")
        print(f"Entrada ONNX: {self.input_name} | dtype={self.input_dtype}")

    def preprocess(self, gray):
        """
        Replica el preprocesamiento oficial:
        - resize a 640 x 480
        - escala [0, 1]
        - duplicación a 3 canales
        - normalización ImageNet
        - formato NCHW
        """
        resized = cv2.resize(
            gray.astype(float),
            (MODEL_WIDTH, MODEL_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        nn_input = resized / 255.0

        nn_input = np.expand_dims(nn_input, axis=-1)
        nn_input = np.tile(nn_input, (1, 1, 3))

        means = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        stds = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        nn_input = nn_input - means
        nn_input = nn_input / stds

        nn_input = nn_input.transpose(2, 0, 1)
        nn_input = np.expand_dims(nn_input, axis=0)

        return nn_input.astype(self.input_dtype)

    def predict_probabilities(self, gray):
        """Retorna mapas de probabilidad en la resolución original."""
        original_height, original_width = gray.shape

        nn_input = self.preprocess(gray)

        output = self.session.run(
            None,
            {self.input_name: nn_input}
        )[0]

        output = np.squeeze(output, axis=0)
        output = np.transpose(output, (1, 2, 0))

        probabilities = cv2.resize(
            output.astype(np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        )

        return probabilities

    def predict_masks(self, gray):
        """Retorna máscaras binarias limpias para cada clase."""
        probabilities = self.predict_probabilities(gray)

        masks = {}

        for class_name, class_index in self.CLASS_MAPPING.items():
            binary = (
                probabilities[:, :, class_index] >=
                SEG_THRESHOLD[class_name]
            ).astype(np.uint8)

            if class_name != "eyelashes":
                binary = largest_component(binary)
                binary = fill_holes(binary)

            masks[class_name] = binary

        return masks, probabilities


# ============================================================
# GEOMETRÍA DEL IRIS
# ============================================================

def estimate_iris_geometry(masks):
    """
    Estima centro y diámetro del iris a partir de iris + pupil.
    """
    iris_disk = (
        (masks["iris"] > 0) |
        (masks["pupil"] > 0)
    ).astype(np.uint8)

    iris_disk = cv2.morphologyEx(
        iris_disk,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9)
        )
    )

    iris_disk = largest_component(iris_disk)
    iris_disk = fill_holes(iris_disk)

    contours, _ = cv2.findContours(
        iris_disk,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise RuntimeError("No se pudo estimar la geometría del iris.")

    contour = max(contours, key=cv2.contourArea)

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)

        (cx, cy), (diameter_a, diameter_b), angle = ellipse

        diameter_px = float(
            (diameter_a + diameter_b) / 2.0
        )

    else:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        diameter_px = float(2.0 * radius)
        angle = 0.0

    if diameter_px < 30:
        raise RuntimeError(
            f"Diámetro de iris no válido: {diameter_px:.2f} px."
        )

    return {
        "iris_disk": iris_disk,
        "contour": contour,
        "cx": float(cx),
        "cy": float(cy),
        "diameter_px": diameter_px,
        "radius_px": diameter_px / 2.0,
        "angle": float(angle)
    }


# ============================================================
# BORDE PALPEBRAL INFERIOR
# ============================================================

def estimate_lower_lid_curve(masks, geometry):
    """
    Estima la curva inferior de la región ocular visible.

    Esta curva se usa como aproximación del límite inferior
    del menisco lagrimal.
    """
    eye_visible = (
        (masks["eyeball"] > 0) |
        (masks["iris"] > 0) |
        (masks["pupil"] > 0)
    ).astype(np.uint8)

    eye_visible = cv2.morphologyEx(
        eye_visible,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11)
        )
    )

    eye_visible = largest_component(eye_visible)
    eye_visible = fill_holes(eye_visible)

    height, width = eye_visible.shape

    lower_curve = np.full(
        width,
        np.nan,
        dtype=np.float32
    )

    for x in range(width):
        ys = np.flatnonzero(eye_visible[:, x] > 0)

        if len(ys) > 0:
            lower_curve[x] = float(ys.max())

    valid_x = np.flatnonzero(~np.isnan(lower_curve))

    if len(valid_x) < 40:
        raise RuntimeError(
            "La máscara ocular no permite obtener el borde inferior."
        )

    eye_left = int(valid_x.min())
    eye_right = int(valid_x.max())

    continuous_x = np.arange(
        eye_left,
        eye_right + 1
    )

    lower_curve[continuous_x] = np.interp(
        continuous_x,
        valid_x,
        lower_curve[valid_x]
    )

    lower_curve[continuous_x] = smooth_1d(
        lower_curve[continuous_x],
        kernel_size=max(9, 0.045 * len(continuous_x))
    )

    cx = geometry["cx"]
    radius = geometry["radius_px"]

    # Zona inferior central: evita cantos y extremos laterales.
    x0 = max(
        eye_left,
        int(round(cx - 1.18 * radius))
    )

    x1 = min(
        eye_right,
        int(round(cx + 1.18 * radius))
    )

    if x1 - x0 < 30:
        raise RuntimeError(
            "La región útil inferior del ojo es demasiado estrecha."
        )

    return {
        "eye_visible": eye_visible,
        "lower_curve": lower_curve,
        "eye_left": eye_left,
        "eye_right": eye_right,
        "x0": x0,
        "x1": x1
    }


# ============================================================
# DETECCIÓN DEL REFLEJO ESPECULAR EN LA PUPILA
# ============================================================

def detect_pupil_reflection(gray, masks, geometry):
    """
    Detecta el reflejo especular brillante dentro de la pupila.
    Su centro X define la línea vertical usada para localizar
    el reflejo del menisco, según la figura 3 del artículo.
    """
    pupil_mask = masks["pupil"]

    if int(pupil_mask.sum()) == 0:
        return {"cx": geometry["cx"], "cy": geometry["cy"], "found": False}

    enhanced = enhance_gray(gray)
    diameter = geometry["diameter_px"]

    kernel_size = odd(max(7, 0.06 * diameter))

    top_hat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
    )

    roi_values = top_hat[pupil_mask > 0]

    if len(roi_values) == 0 or float(roi_values.max()) == 0:
        return {"cx": geometry["cx"], "cy": geometry["cy"], "found": False}

    threshold = max(
        float(np.percentile(roi_values, 95)),
        float(roi_values.mean() + 1.5 * roi_values.std())
    )

    bright_mask = (
        (pupil_mask > 0) & (top_hat >= threshold)
    ).astype(np.uint8)

    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    number, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8
    )

    if number <= 1:
        return {"cx": geometry["cx"], "cy": geometry["cy"], "found": False}

    min_area = max(2, int(round(0.00005 * diameter ** 2)))
    max_area = max(30, int(round(0.012 * diameter ** 2)))

    best_id = None
    best_val = -1.0

    for cid in range(1, number):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        comp = labels == cid
        mean_val = float(top_hat[comp].mean())
        if mean_val > best_val:
            best_val = mean_val
            best_id = cid

    if best_id is None:
        return {"cx": geometry["cx"], "cy": geometry["cy"], "found": False}

    return {
        "cx": float(centroids[best_id][0]),
        "cy": float(centroids[best_id][1]),
        "found": True,
        "area": int(stats[best_id, cv2.CC_STAT_AREA]),
        "score": best_val
    }


# ============================================================
# ESCALA Y RANGO DE TMH
# ============================================================

def calculate_mm_per_pixel(geometry):
    """
    Prioridad:
    1. MM_PER_PIXEL calibrado físicamente.
    2. Conversión aproximada usando diámetro del iris.
    """
    if MM_PER_PIXEL is not None:
        return float(MM_PER_PIXEL), "calibracion_fisica"

    if REFERENCE_IRIS_DIAMETER_MM is not None:
        scale = (
            float(REFERENCE_IRIS_DIAMETER_MM) /
            float(geometry["diameter_px"])
        )

        return scale, "diametro_iris_aproximado"

    return None, "sin_conversion_mm"


def calculate_tmh_limits_px(geometry):
    """Obtiene límites de búsqueda vertical del TMH en píxeles."""
    mm_per_pixel, _ = calculate_mm_per_pixel(geometry)

    if mm_per_pixel is not None:
        min_height = max(
            2,
            int(round(TMH_MIN_MM / mm_per_pixel))
        )

        max_height = max(
            min_height + 4,
            int(round(TMH_MAX_MM / mm_per_pixel))
        )

    else:
        diameter = geometry["diameter_px"]

        min_height = max(
            2,
            int(round(0.005 * diameter))
        )

        max_height = max(
            min_height + 4,
            int(round(0.07 * diameter))
        )

    return min_height, max_height


# ============================================================
# ROI GENERAL DEL MENISCO
# ============================================================

def build_meniscus_roi(gray, masks, geometry, lid_info):
    """
    Construye una ROI que sigue el borde palpebral inferior.
    No utiliza coordenadas fijas ni el reflejo pupilar.
    """
    image_height, image_width = gray.shape

    lower_curve = lid_info["lower_curve"]

    min_height, max_height = calculate_tmh_limits_px(geometry)

    # Margen superior suficiente para cubrir el menisco.
    upper_band = max_height + max(
        4,
        int(round(0.03 * geometry["diameter_px"]))
    )

    # Pequeña tolerancia debajo de la curva inferior.
    lower_band = max(
        3,
        int(round(0.02 * geometry["diameter_px"]))
    )

    roi = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for x in range(lid_info["x0"], lid_info["x1"] + 1):
        if np.isnan(lower_curve[x]):
            continue

        y_lower = int(round(lower_curve[x]))

        y_start = max(0, y_lower - upper_band)
        y_end = min(
            image_height,
            y_lower + lower_band + 1
        )

        roi[y_start:y_end, x] = 1

    # Se excluye el interior profundo del iris para no elegir
    # el reflejo pupilar, pero se conserva la franja próxima al borde.
    iris_erosion = odd(
        max(5, 0.13 * geometry["diameter_px"])
    )

    iris_core = cv2.erode(
        geometry["iris_disk"],
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (iris_erosion, iris_erosion)
        )
    )

    roi[iris_core > 0] = 0

    # Excluir pestañas detectadas y su vecindario inmediato.
    eyelash_dilation = odd(
        max(3, 0.025 * geometry["diameter_px"])
    )

    eyelashes_zone = cv2.dilate(
        masks["eyelashes"],
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (eyelash_dilation, eyelash_dilation)
        )
    )

    roi[eyelashes_zone > 0] = 0

    return roi


# ============================================================
# DETECCIÓN DEL REFLEJO ESPECULAR DEL MENISCO
# ============================================================

def detect_meniscus_reflection(
    gray,
    roi,
    lid_info,
    geometry,
    vertical_line_x=None,
    intersection_y=None
):
    """
    Detecta candidatos especulares en toda la ROI del menisco.

    Score:
    - intensidad top-hat;
    - brillo local;
    - proximidad al borde inferior;
    - área y forma razonables.

    No asume si el reflejo está a la derecha o izquierda.
    """
    enhanced = enhance_gray(gray)

    diameter = geometry["diameter_px"]

    kernel_size = odd(
        max(9, 0.075 * diameter)
    )

    top_hat = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
    )

    roi_values_top = top_hat[roi > 0]
    roi_values_gray = enhanced[roi > 0]

    if len(roi_values_top) == 0:
        return None, pd.DataFrame(), np.zeros_like(gray), top_hat, np.zeros_like(gray)

    top_threshold = max(
        float(np.percentile(roi_values_top, 96.0)),
        float(roi_values_top.mean() + 1.15 * roi_values_top.std())
    )

    intensity_threshold = float(
        np.percentile(roi_values_gray, 78.0)
    )

    all_candidates_mask = (
        (roi > 0) &
        (top_hat >= top_threshold) &
        (enhanced >= intensity_threshold)
    ).astype(np.uint8)

    all_candidates_mask = cv2.morphologyEx(
        all_candidates_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
    )

    number, labels, stats, centroids = cv2.connectedComponentsWithStats(
        all_candidates_mask,
        connectivity=8
    )

    lower_curve = lid_info["lower_curve"]

    max_top_hat = max(
        1.0,
        float(roi_values_top.max())
    )

    diameter_squared = diameter ** 2

    min_area = max(
        2,
        int(round(0.00002 * diameter_squared))
    )

    max_area = max(
        25,
        int(round(0.018 * diameter_squared))
    )

    min_height_px, _ = calculate_tmh_limits_px(geometry)

    rows = []

    for component_id in range(1, number):
        x, y, w, h, area = stats[component_id]

        if area < min_area or area > max_area:
            continue

        component = labels == component_id

        center_x, center_y = centroids[component_id]

        x_index = int(
            np.clip(round(center_x), 0, gray.shape[1] - 1)
        )

        if np.isnan(lower_curve[x_index]):
            continue

        distance_to_lower_border = float(
            lower_curve[x_index] - center_y
        )

        # El reflejo debe estar al menos min_height_px encima del borde,
        # lo que garantiza un TMH mínimo fisiológicamente válido.
        if distance_to_lower_border < min_height_px:
            continue

        mean_top_hat = float(
            top_hat[component].mean()
        )

        max_intensity = float(
            enhanced[component].max()
        )

        aspect_ratio = float(
            w / max(h, 1)
        )

        shape_score = float(
            np.exp(
                -abs(np.log(max(aspect_ratio, 1e-6))) / 2.2
            )
        )

        proximity_score = float(
            np.exp(
                -abs(distance_to_lower_border) /
                max(3.0, 0.055 * diameter)
            )
        )

        score = (
            0.42 * (mean_top_hat / max_top_hat) +
            0.20 * (max_intensity / 255.0) +
            0.28 * proximity_score +
            0.10 * shape_score
        )

        rows.append({
            "id": int(component_id),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "cx": float(center_x),
            "cy": float(center_y),
            "distance_to_lower_border_px": distance_to_lower_border,
            "mean_top_hat": mean_top_hat,
            "max_intensity": max_intensity,
            "aspect_ratio": aspect_ratio,
            "score": float(score)
        })

    if not rows:
        return (
            None,
            pd.DataFrame(),
            np.zeros_like(gray),
            top_hat,
            all_candidates_mask * 255
        )

    candidates_df = pd.DataFrame(rows)

    if vertical_line_x is not None and intersection_y is not None:
        # Método del artículo: seleccionar el reflejo más cercano al punto
        # de intersección de la línea vertical (centro del reflejo pupilar)
        # con el borde del párpado inferior.
        cx_arr = candidates_df["cx"].values
        cy_arr = candidates_df["cy"].values
        distances = np.sqrt(
            (cx_arr - float(vertical_line_x)) ** 2 +
            (cy_arr - float(intersection_y)) ** 2
        )
        best_row = candidates_df.iloc[int(np.argmin(distances))].to_dict()
    else:
        best_row = (
            candidates_df
            .sort_values("score", ascending=False)
            .iloc[0]
            .to_dict()
        )

    candidates = (
        candidates_df
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    best = {
        key: json_serializable(value)
        for key, value in best_row.items()
    }

    selected_mask = (
        labels == int(best["id"])
    ).astype(np.uint8) * 255

    return (
        best,
        candidates,
        selected_mask,
        top_hat,
        all_candidates_mask * 255
    )


# ============================================================
# CÁLCULO DE TMH POR TRAZADO CONTINUO DEL BORDE SUPERIOR
# ============================================================

def estimate_tmh(
    gray,
    geometry,
    lid_info,
    reflection=None,
    reflection_mask=None
):
    """
    Estima TMH como distancia vertical entre:

    - borde inferior: curva inferior de la región ocular visible;
    - borde superior: trayectoria continua de alto gradiente,
      detectada mediante programación dinámica.

    El reflejo detectado se elimina temporalmente con inpainting
    para evitar medir su propio contorno como límite del menisco.
    """
    min_height, max_height = calculate_tmh_limits_px(geometry)

    diameter = geometry["diameter_px"]

    # Ventana de medición:
    # centrada en el reflejo si existe; de lo contrario, zona central.
    if reflection is not None:
        center_x = float(reflection["cx"])

        half_window = max(
            int(round(0.15 * diameter)),
            int(round(2.5 * reflection["w"]))
        )

        measurement_mode = "centrado_en_reflejo_detectado"

    else:
        center_x = float(geometry["cx"])

        half_window = int(round(0.22 * diameter))

        measurement_mode = "zona_central_sin_reflejo"

    x0 = max(
        lid_info["x0"],
        int(round(center_x - half_window))
    )

    x1 = min(
        lid_info["x1"],
        int(round(center_x + half_window))
    )

    x_values = np.arange(x0, x1 + 1)

    if len(x_values) < 12:
        raise RuntimeError(
            "No hay suficientes columnas válidas para estimar TMH."
        )

    enhanced = enhance_gray(gray)

    # Eliminar temporalmente el brillo especular seleccionado.
    if reflection_mask is not None and np.any(reflection_mask > 0):
        inpaint_size = odd(
            max(5, 0.022 * diameter)
        )

        inpaint_mask = cv2.dilate(
            reflection_mask.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (inpaint_size, inpaint_size)
            )
        )

        enhanced_without_reflection = cv2.inpaint(
            enhanced,
            inpaint_mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )

    else:
        enhanced_without_reflection = enhanced.copy()

    gradient_vertical = np.abs(
        cv2.Sobel(
            enhanced_without_reflection,
            cv2.CV_32F,
            dx=0,
            dy=1,
            ksize=3
        )
    )

    gradient_vertical = cv2.GaussianBlur(
        gradient_vertical,
        (3, 3),
        0
    )

    candidate_heights = np.arange(
        min_height,
        max_height + 1
    )

    number_columns = len(x_values)
    number_heights = len(candidate_heights)

    lower_y = np.zeros(
        number_columns,
        dtype=np.int32
    )

    energy = np.zeros(
        (number_columns, number_heights),
        dtype=np.float32
    )

    for column_index, x in enumerate(x_values):
        lower_y[column_index] = int(
            round(lid_info["lower_curve"][x])
        )

        possible_upper_y = (
            lower_y[column_index] -
            candidate_heights
        )

        valid = (
            (possible_upper_y >= 0) &
            (possible_upper_y < gray.shape[0])
        )

        energy[column_index, valid] = gradient_vertical[
            possible_upper_y[valid],
            x
        ]

    p5, p99 = np.percentile(energy, [5, 99])

    normalized_energy = np.clip(
        (energy - p5) / max(p99 - p5, 1e-6),
        0,
        1
    )

    # Programación dinámica:
    # maximiza el gradiente evitando saltos bruscos entre columnas.
    smoothness_penalty = 0.11

    dp = np.full_like(
        normalized_energy,
        -1e9,
        dtype=np.float32
    )

    predecessor = np.zeros(
        (number_columns, number_heights),
        dtype=np.int32
    )

    dp[0, :] = normalized_energy[0, :]

    height_indices = np.arange(number_heights)

    for i in range(1, number_columns):
        transitions = (
            dp[i - 1, :, None] -
            smoothness_penalty *
            np.abs(
                height_indices[:, None] -
                height_indices[None, :]
            )
        )

        predecessor[i, :] = np.argmax(
            transitions,
            axis=0
        )

        dp[i, :] = (
            normalized_energy[i, :] +
            transitions[
                predecessor[i, :],
                height_indices
            ]
        )

    path = np.zeros(
        number_columns,
        dtype=np.int32
    )

    path[-1] = int(
        np.argmax(dp[-1, :])
    )

    for i in range(number_columns - 1, 0, -1):
        path[i - 1] = predecessor[i, path[i]]

    tmh_column_px = candidate_heights[path]

    upper_y = lower_y - tmh_column_px

    edge_strength = normalized_energy[
        np.arange(number_columns),
        path
    ]

    strength_threshold = float(
        np.percentile(edge_strength, 20)
    )

    reliable_columns = (
        edge_strength >= strength_threshold
    )

    minimum_reliable_columns = max(
        5,
        int(round(0.30 * number_columns))
    )

    if reliable_columns.sum() < minimum_reliable_columns:
        reliable_columns[:] = True

    valid_tmh = tmh_column_px[
        reliable_columns
    ]

    tmh_px = float(
        np.median(valid_tmh)
    )

    q1_px = float(
        np.percentile(valid_tmh, 25)
    )

    q3_px = float(
        np.percentile(valid_tmh, 75)
    )

    fraction_at_limits = float(
        np.mean(
            (tmh_column_px == min_height) |
            (tmh_column_px == max_height)
        )
    )

    median_edge_strength = float(
        np.median(edge_strength)
    )

    if (
        fraction_at_limits <= 0.25 and
        median_edge_strength >= 0.20
    ):
        quality = "OK"

    elif fraction_at_limits <= 0.45:
        quality = "REVISAR"

    else:
        quality = "NO_CONFIABLE"

    mm_per_pixel, scale_source = calculate_mm_per_pixel(
        geometry
    )

    tmh_mm = None
    q1_mm = None
    q3_mm = None

    if mm_per_pixel is not None:
        tmh_mm = float(tmh_px * mm_per_pixel)
        q1_mm = float(q1_px * mm_per_pixel)
        q3_mm = float(q3_px * mm_per_pixel)

    return {
        "x_values": x_values,
        "upper_y": upper_y,
        "lower_y": lower_y,
        "tmh_column_px": tmh_column_px,
        "edge_strength": edge_strength,
        "reliable_columns": reliable_columns,
        "minimum_height_px": int(min_height),
        "maximum_height_px": int(max_height),
        "tmh_px": tmh_px,
        "tmh_q1_px": q1_px,
        "tmh_q3_px": q3_px,
        "tmh_iqr_px": float(q3_px - q1_px),
        "mm_per_pixel": mm_per_pixel,
        "scale_source": scale_source,
        "tmh_mm": tmh_mm,
        "tmh_q1_mm": q1_mm,
        "tmh_q3_mm": q3_mm,
        "fraction_at_limits": fraction_at_limits,
        "median_edge_strength": median_edge_strength,
        "quality": quality,
        "measurement_mode": measurement_mode
    }


# ============================================================
# TMH BASADO EN REFLEJO DEL MENISCO (MÉTODO DEL ARTÍCULO)
# ============================================================

def calculate_tmh_from_reflection(reflection, lid_info, geometry):
    """
    Calcula el TMH según el método del artículo (figura 3):
    distancia vertical entre el centro del reflejo del menisco
    y el borde del párpado inferior en la misma columna X.
    """
    cx = reflection["cx"]
    cy = reflection["cy"]
    lower_curve = lid_info["lower_curve"]

    x_idx = int(np.clip(round(cx), 0, len(lower_curve) - 1))

    if np.isnan(lower_curve[x_idx]):
        valid_x = np.flatnonzero(~np.isnan(lower_curve))
        if len(valid_x) == 0:
            raise RuntimeError(
                "No hay borde inferior válido para calcular TMH."
            )
        x_idx = int(valid_x[np.argmin(np.abs(valid_x - x_idx))])

    lower_y = float(lower_curve[x_idx])
    tmh_px = max(0.0, lower_y - float(cy))

    # Fórmula del artículo: TMH_mm = 11.5 × TMH_px / iris_diameter_px
    iris_diameter_px = geometry["diameter_px"]
    tmh_mm = 11.5 * tmh_px / iris_diameter_px
    mm_per_pixel = 11.5 / iris_diameter_px

    return {
        "tmh_px": tmh_px,
        "tmh_mm": tmh_mm,
        "mm_per_pixel": mm_per_pixel,
        "scale_source": "formula_articulo_11.5/iris_diameter",
        "lower_y_at_reflection": lower_y,
        "reflection_cy": float(cy),
        "measurement_x": x_idx
    }


# ============================================================
# VISUALIZACIONES
# ============================================================

def build_segmentation_visualization(gray, masks):
    """Superpone las máscaras semánticas sobre la imagen."""
    base = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    overlay = base.copy()

    # BGR
    overlay[masks["eyeball"] > 0] = (255, 180, 0)
    overlay[masks["iris"] > 0] = (0, 255, 0)
    overlay[masks["pupil"] > 0] = (0, 0, 255)
    overlay[masks["eyelashes"] > 0] = (255, 0, 255)

    return cv2.addWeighted(
        base,
        0.70,
        overlay,
        0.30,
        0
    )


def build_roi_visualization(gray, roi, lid_info):
    """Visualiza la ROI y la curva palpebral inferior."""
    result = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    overlay = result.copy()
    overlay[roi > 0] = (0, 200, 255)

    result = cv2.addWeighted(
        result,
        0.78,
        overlay,
        0.22,
        0
    )

    xs = np.arange(
        lid_info["x0"],
        lid_info["x1"] + 1
    )

    ys = np.array([
        int(round(lid_info["lower_curve"][x]))
        for x in xs
    ])

    points = np.column_stack([
        xs,
        ys
    ]).astype(np.int32)

    cv2.polylines(
        result,
        [points],
        False,
        (255, 0, 0),
        2
    )

    return result


def draw_final_result(
    gray,
    geometry,
    roi,
    reflection,
    all_candidates_mask,
    tmh_info,
    pupil_reflection=None,
    vertical_line_x=None,
    intersection_y=None
):
    """Dibuja el resultado completo sobre la imagen."""
    result = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    overlay = result.copy()
    overlay[roi > 0] = (0, 200, 255)

    result = cv2.addWeighted(
        result,
        0.84,
        overlay,
        0.16,
        0
    )

    # Contorno del iris.
    cv2.drawContours(
        result,
        [geometry["contour"]],
        -1,
        (255, 180, 0),
        1
    )

    # Candidatos de reflejo en naranja.
    candidate_contours, _ = cv2.findContours(
        all_candidates_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        result,
        candidate_contours,
        -1,
        (0, 165, 255),
        1
    )

    # Bordes empleados para TMH.
    x_values = tmh_info["x_values"]

    upper_points = np.column_stack([
        x_values,
        tmh_info["upper_y"]
    ]).astype(np.int32)

    lower_points = np.column_stack([
        x_values,
        tmh_info["lower_y"]
    ]).astype(np.int32)

    # Rojo: borde superior del menisco.
    cv2.polylines(
        result,
        [upper_points],
        False,
        (0, 0, 255),
        2
    )

    # Azul: borde inferior del menisco.
    cv2.polylines(
        result,
        [lower_points],
        False,
        (255, 0, 0),
        2
    )

    # Reflejo seleccionado.
    if reflection is not None:
        x = int(reflection["x"])
        y = int(reflection["y"])
        w = int(reflection["w"])
        h = int(reflection["h"])

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.circle(
            result,
            (
                int(round(reflection["cx"])),
                int(round(reflection["cy"]))
            ),
            3,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            result,
            "Reflejo menisco",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    # Reflejo pupilar y línea vertical (cyan).
    if pupil_reflection is not None and pupil_reflection.get("found"):
        pup_cx = int(round(pupil_reflection["cx"]))
        pup_cy = int(round(pupil_reflection["cy"]))

        cv2.circle(
            result,
            (pup_cx, pup_cy),
            5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            result,
            "Reflejo pupila",
            (pup_cx + 7, pup_cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    if vertical_line_x is not None and intersection_y is not None:
        int_y = int(round(intersection_y))
        top_y = (
            int(round(pupil_reflection["cy"]))
            if (pupil_reflection is not None and pupil_reflection.get("found"))
            else 0
        )

        # Línea vertical desde el reflejo pupilar hasta la intersección.
        cv2.line(
            result,
            (vertical_line_x, top_y),
            (vertical_line_x, int_y),
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Punto de intersección.
        cv2.circle(
            result,
            (vertical_line_x, int_y),
            4,
            (0, 255, 255),
            -1
        )

    # Segmento vertical del TMH: desde el reflejo del menisco
    # hasta el borde inferior (medición del artículo).
    if reflection is not None:
        cx_r = int(round(reflection["cx"]))
        cy_r = int(round(reflection["cy"]))
        lower_y_r = int(round(cy_r + tmh_info["tmh_px"]))

        cv2.line(
            result,
            (cx_r, cy_r),
            (cx_r, lower_y_r),
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Ticks horizontales en los extremos del segmento TMH.
        for tick_y in (cy_r, lower_y_r):
            cv2.line(
                result,
                (cx_r - 4, tick_y),
                (cx_r + 4, tick_y),
                (255, 255, 0),
                1
            )

    label = f"TMH={tmh_info['tmh_px']:.2f} px"

    if tmh_info["tmh_mm"] is not None:
        label += f" | {tmh_info['tmh_mm']:.3f} mm aprox."

    cv2.putText(
        result,
        label,
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        result,
        f"Calidad: {tmh_info['quality']}",
        (15, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    return result


# ============================================================
# PROCESAMIENTO DE UNA IMAGEN
# ============================================================

def process_image(image_path, segmenter, output_root):
    image_path = Path(image_path)

    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise FileNotFoundError(
            f"No se pudo abrir la imagen: {image_path}"
        )

    image_output_dir = (
        Path(output_root) /
        image_path.stem
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Segmentación ocular
    # --------------------------------------------------------
    masks, probabilities = segmenter.predict_masks(gray)

    # --------------------------------------------------------
    # 2. Geometría del iris
    # --------------------------------------------------------
    geometry = estimate_iris_geometry(masks)

    # --------------------------------------------------------
    # 3. Borde inferior del ojo / menisco
    # --------------------------------------------------------
    lid_info = estimate_lower_lid_curve(
        masks,
        geometry
    )

    # --------------------------------------------------------
    # 3b. Reflejo especular en la pupila
    #     Su centro X define la línea vertical del artículo.
    # --------------------------------------------------------
    pupil_reflection = detect_pupil_reflection(gray, masks, geometry)

    vertical_line_x = int(np.clip(
        round(pupil_reflection["cx"]),
        lid_info["x0"],
        lid_info["x1"]
    ))

    if np.isnan(lid_info["lower_curve"][vertical_line_x]):
        valid_xs = np.flatnonzero(
            ~np.isnan(lid_info["lower_curve"])
        )
        if len(valid_xs) > 0:
            vertical_line_x = int(
                valid_xs[np.argmin(np.abs(valid_xs - vertical_line_x))]
            )

    intersection_y = float(lid_info["lower_curve"][vertical_line_x])

    # --------------------------------------------------------
    # 4. ROI general del menisco
    # --------------------------------------------------------
    roi = build_meniscus_roi(
        gray,
        masks,
        geometry,
        lid_info
    )

    # --------------------------------------------------------
    # 5. Reflejo especular del menisco
    #    Se selecciona el más cercano al punto de intersección
    #    de la línea vertical con el borde del párpado inferior.
    # --------------------------------------------------------
    (
        reflection,
        candidates,
        reflection_mask,
        top_hat,
        all_candidates_mask
    ) = detect_meniscus_reflection(
        gray,
        roi,
        lid_info,
        geometry,
        vertical_line_x=vertical_line_x,
        intersection_y=intersection_y
    )

    # --------------------------------------------------------
    # 5b. TMH basado en el reflejo (método del artículo):
    #     distancia vertical entre el reflejo del menisco
    #     y el borde del párpado inferior.
    # --------------------------------------------------------
    if reflection is not None:
        tmh_from_reflection = calculate_tmh_from_reflection(
            reflection,
            lid_info,
            geometry
        )
    else:
        tmh_from_reflection = None

    # --------------------------------------------------------
    # 6. TMH por trazado continuo de borde (para visualización)
    # --------------------------------------------------------
    tmh_info = estimate_tmh(
        gray,
        geometry,
        lid_info,
        reflection=reflection,
        reflection_mask=reflection_mask
    )

    # El TMH primario es el del artículo cuando el reflejo fue detectado.
    if tmh_from_reflection is not None:
        tmh_info["tmh_px"] = tmh_from_reflection["tmh_px"]
        tmh_info["tmh_mm"] = tmh_from_reflection["tmh_mm"]
        tmh_info["mm_per_pixel"] = tmh_from_reflection["mm_per_pixel"]
        tmh_info["scale_source"] = tmh_from_reflection["scale_source"]
        tmh_info["measurement_mode"] = "reflejo_menisco_vertical"

    # --------------------------------------------------------
    # 7. Visualizaciones
    # --------------------------------------------------------
    segmentation_vis = build_segmentation_visualization(
        gray,
        masks
    )

    roi_vis = build_roi_visualization(
        gray,
        roi,
        lid_info
    )

    final_vis = draw_final_result(
        gray,
        geometry,
        roi,
        reflection,
        all_candidates_mask,
        tmh_info,
        pupil_reflection=pupil_reflection,
        vertical_line_x=vertical_line_x,
        intersection_y=intersection_y
    )

    # --------------------------------------------------------
    # 8. Guardado de imágenes
    # --------------------------------------------------------
    if SAVE_INTERMEDIATE_IMAGES:
        cv2.imwrite(
            str(image_output_dir / "01_imagen_original.png"),
            gray
        )

        cv2.imwrite(
            str(image_output_dir / "02_segmentacion.png"),
            segmentation_vis
        )

        cv2.imwrite(
            str(image_output_dir / "03_roi_menisco.png"),
            roi * 255
        )

        cv2.imwrite(
            str(image_output_dir / "04_roi_sobre_imagen.png"),
            roi_vis
        )

        cv2.imwrite(
            str(image_output_dir / "05_top_hat.png"),
            top_hat
        )

        cv2.imwrite(
            str(image_output_dir / "06_todos_los_candidatos.png"),
            all_candidates_mask
        )

        cv2.imwrite(
            str(image_output_dir / "07_reflejo_seleccionado.png"),
            reflection_mask
        )

        for mask_name, mask in masks.items():
            cv2.imwrite(
                str(image_output_dir / f"mask_{mask_name}.png"),
                mask * 255
            )

    cv2.imwrite(
        str(image_output_dir / "08_resultado_tmh.png"),
        final_vis
    )

    # --------------------------------------------------------
    # 9. Guardado de tablas
    # --------------------------------------------------------
    candidates.to_csv(
        image_output_dir / "candidatos_reflejo.csv",
        index=False
    )

    profile = pd.DataFrame({
        "x": tmh_info["x_values"],
        "upper_boundary_y": tmh_info["upper_y"],
        "lower_boundary_y": tmh_info["lower_y"],
        "tmh_column_px": tmh_info["tmh_column_px"],
        "edge_strength": tmh_info["edge_strength"],
        "reliable_column": tmh_info["reliable_columns"]
    })

    profile.to_csv(
        image_output_dir / "perfil_tmh_por_columna.csv",
        index=False
    )

    # --------------------------------------------------------
    # 10. Resultado numérico
    # --------------------------------------------------------
    result = {
        "image": image_path.name,
        "iris_center_x_px": geometry["cx"],
        "iris_center_y_px": geometry["cy"],
        "iris_diameter_px": geometry["diameter_px"],
        "pupil_reflection_found": pupil_reflection["found"],
        "pupil_reflection_x_px": (
            pupil_reflection["cx"] if pupil_reflection["found"] else None
        ),
        "pupil_reflection_y_px": (
            pupil_reflection["cy"] if pupil_reflection["found"] else None
        ),
        "vertical_line_x": vertical_line_x,
        "intersection_y": intersection_y,
        "reflection_detected": reflection is not None,
        "reflection_x_px": (
            None if reflection is None else reflection["cx"]
        ),
        "reflection_y_px": (
            None if reflection is None else reflection["cy"]
        ),
        "reflection_score": (
            None if reflection is None else reflection["score"]
        ),
        "tmh_px": tmh_info["tmh_px"],
        "tmh_q1_px": tmh_info["tmh_q1_px"],
        "tmh_q3_px": tmh_info["tmh_q3_px"],
        "tmh_iqr_px": tmh_info["tmh_iqr_px"],
        "tmh_mm_approx": tmh_info["tmh_mm"],
        "tmh_q1_mm_approx": tmh_info["tmh_q1_mm"],
        "tmh_q3_mm_approx": tmh_info["tmh_q3_mm"],
        "mm_per_pixel": tmh_info["mm_per_pixel"],
        "scale_source": tmh_info["scale_source"],
        "measurement_mode": tmh_info["measurement_mode"],
        "quality": tmh_info["quality"],
        "fraction_at_limits": tmh_info["fraction_at_limits"],
        "median_edge_strength": tmh_info["median_edge_strength"]
    }

    result = {
        key: json_serializable(value)
        for key, value in result.items()
    }

    with open(
        image_output_dir / "resultado.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # 11. Mostrar resultado
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Imagen: {image_path.name}")
    print("=" * 70)

    print(
        f"Iris: centro=({result['iris_center_x_px']:.2f}, "
        f"{result['iris_center_y_px']:.2f}) | "
        f"diámetro={result['iris_diameter_px']:.2f} px"
    )

    if pupil_reflection["found"]:
        print(
            f"Reflejo pupilar: "
            f"({result['pupil_reflection_x_px']:.2f}, "
            f"{result['pupil_reflection_y_px']:.2f}) | "
            f"línea vertical x={vertical_line_x} | "
            f"intersección y={intersection_y:.2f}"
        )
    else:
        print("Reflejo pupilar: no detectado (usando centro del iris)")

    if reflection is None:
        print("Reflejo del menisco: no detectado")
    else:
        print(
            f"Reflejo del menisco: "
            f"({result['reflection_x_px']:.2f}, "
            f"{result['reflection_y_px']:.2f}) | "
            f"score={result['reflection_score']:.4f}"
        )

    print(f"TMH: {result['tmh_px']:.2f} px")

    if result["tmh_mm_approx"] is not None:
        print(
            f"TMH aproximado: {result['tmh_mm_approx']:.4f} mm "
            f"| escala={result['scale_source']}"
        )

    print(f"Calidad automática: {result['quality']}")
    print(f"Archivos: {image_output_dir}")

    if SHOW_PLOTS:
        figure, axes = plt.subplots(
            1,
            5,
            figsize=(27, 6)
        )

        axes[0].imshow(gray, cmap="gray")
        axes[0].set_title("Imagen original")

        axes[1].imshow(
            cv2.cvtColor(
                segmentation_vis,
                cv2.COLOR_BGR2RGB
            )
        )
        axes[1].set_title("Segmentación ONNX")

        axes[2].imshow(
            cv2.cvtColor(
                roi_vis,
                cv2.COLOR_BGR2RGB
            )
        )
        axes[2].set_title("ROI del menisco")

        axes[3].imshow(
            all_candidates_mask,
            cmap="gray"
        )
        axes[3].set_title("Candidatos de reflejo")

        axes[4].imshow(
            cv2.cvtColor(
                final_vis,
                cv2.COLOR_BGR2RGB
            )
        )
        axes[4].set_title("Reflejo + TMH")

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()
        plt.show()

    return result, candidates, profile


# ============================================================
# EJECUCIÓN GENERAL
# ============================================================

def run_pipeline():
    output_root = Path(OUTPUT_DIR)

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = resolve_image_paths(
        INPUT_PATH
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No se encontraron imágenes en: {INPUT_PATH}"
        )

    print("=" * 72)
    print("PIPELINE GENERAL: REFLEJO DEL MENISCO + TMH")
    print("=" * 72)
    print(f"Imágenes encontradas: {len(image_paths)}")
    print(f"Salida: {output_root}")

    segmenter = OfficialOpenIrisONNXSegmenter(
        local_model_path=LOCAL_MODEL_PATH
    )

    summary = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):
        print("\n" + "-" * 72)
        print(
            f"[{index}/{len(image_paths)}] "
            f"Procesando: {image_path.name}"
        )

        try:
            result, candidates, profile = process_image(
                image_path,
                segmenter,
                output_root
            )

            summary.append(result)

        except Exception as error:
            warnings.warn(
                f"No se pudo procesar {image_path.name}: {error}"
            )

            summary.append({
                "image": image_path.name,
                "error": str(error)
            })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        output_root / "resumen_resultados_tmh.csv",
        index=False
    )

    print("\n" + "=" * 72)
    print("RESUMEN FINAL")
    print("=" * 72)

    display(summary_df)

    print(f"Resultados guardados en: {output_root}")

    return summary_df


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":
    resultados = run_pipeline()

