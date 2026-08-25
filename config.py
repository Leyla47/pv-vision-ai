"""PV Vision AI merkezi proje ayarları."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
TRAINING_DIR = PROJECT_ROOT / "training"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"

MODEL_WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
MODEL_METADATA_PATH = WEIGHTS_DIR / "model_info.json"
DATASET_YAML_PATH = PROCESSED_DATA_DIR / "dataset.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFECT_CLASSES_TECHNICAL = [
    "crack",
    "finger",
    "black_core",
    "thick_line",
    "star_crack",
    "corner",
    "fragment",
    "scratch",
    "horizontal_dislocation",
    "vertical_dislocation",
    "printing_error",
    "short_circuit",
]

DEFECT_NAME_TR_BY_TECHNICAL = {
    "finger": "Parmak izi kusuru",
    "crack": "Çatlak",
    "black_core": "Siyah çekirdek",
    "thick_line": "Kalın çizgi",
    "horizontal_dislocation": "Yatay hizasızlık",
    "short_circuit": "Kısa devre",
    "vertical_dislocation": "Dikey hizasızlık",
    "star_crack": "Yıldız çatlağı",
    "printing_error": "Baskı hatası",
    "corner": "Köşe kusuru",
    "fragment": "Parçalanma",
    "scratch": "Çizik",
}

DEFECT_CLASSES_TR = [
    DEFECT_NAME_TR_BY_TECHNICAL[class_name]
    for class_name in DEFECT_CLASSES_TECHNICAL
]

CLASS_ID_TO_NAME_TECHNICAL = {
    class_id: class_name for class_id, class_name in enumerate(DEFECT_CLASSES_TECHNICAL)
}

NAME_TECHNICAL_TO_CLASS_ID = {
    class_name: class_id for class_id, class_name in CLASS_ID_TO_NAME_TECHNICAL.items()
}

CLASS_ID_TO_NAME_TR = {
    class_id: DEFECT_NAME_TR_BY_TECHNICAL[class_name]
    for class_id, class_name in CLASS_ID_TO_NAME_TECHNICAL.items()
}

NAME_TR_TO_CLASS_ID = {
    class_name: class_id for class_id, class_name in CLASS_ID_TO_NAME_TR.items()
}

NUM_CLASSES = len(DEFECT_CLASSES_TECHNICAL)

# İlk V1 kalite puanı için açıklanabilir kusur önem katsayıları.
DEFECT_SEVERITY_BY_TECHNICAL = {
    "crack": 3.0,
    "finger": 1.0,
    "black_core": 2.2,
    "thick_line": 1.4,
    "star_crack": 3.0,
    "corner": 1.8,
    "fragment": 3.0,
    "scratch": 1.2,
    "horizontal_dislocation": 2.0,
    "vertical_dislocation": 2.0,
    "printing_error": 1.5,
    "short_circuit": 3.0,
}

DEFECT_SEVERITY_BY_NAME_TR = {
    DEFECT_NAME_TR_BY_TECHNICAL[class_name]: DEFECT_SEVERITY_BY_TECHNICAL[class_name]
    for class_name in DEFECT_CLASSES_TECHNICAL
}

QUALITY_GRADE_THRESHOLDS = {"A": 85.0, "B": 60.0, "C": 0.0}
QUALITY_GRADE_LABELS = {"A": "İyi", "B": "Orta", "C": "Düşük kalite"}
QUALITY_PRICE_COEFFICIENTS = {"A": 0.95, "B": 0.75, "C": 0.45}
SUPPORTED_PRICE_CURRENCIES = ("TRY", "USD", "EUR")
DEFAULT_REFERENCE_PRICE = 1000.0
MIN_REFERENCE_PRICE = 0.01
REFERENCE_PRICE_STEP = 100.0

# Görüntü tabanlı V1 üretim performansı tahmini için kalibre edilebilir ayarlar.
PERFORMANCE_QUALITY_LOSS_FACTOR = 0.30
DEFAULT_REFERENCE_POWER_W = 550
MIN_REFERENCE_POWER_W = 1
MAX_REFERENCE_POWER_W = 5000

# Açıklanabilir V1 panel sağlık skoru ve risk eşikleri.
HEALTH_QUALITY_WEIGHT = 0.60
HEALTH_AREA_WEIGHT = 0.20
HEALTH_AREA_PENALTY_CAP = 20.0
HEALTH_CLASS_EFFECT_FACTOR = 1.50
HEALTH_CLASS_PENALTY_CAP = 20.0
HEALTH_BREAKDOWN_CLASS_LIMIT = 3
HEALTH_STATUS_THRESHOLDS = {
    "Çok İyi": 85.0,
    "İyi": 70.0,
    "Orta": 50.0,
    "Kritik": 0.0,
}
HEALTH_HIGH_RISK_SCORE = 50.0
HEALTH_MEDIUM_RISK_SCORE = 70.0
HEALTH_HIGH_RISK_CRITICAL_COUNT = 2
HEALTH_HIGH_RISK_CRITICAL_AREA_PERCENT = 10.0
CRITICAL_DEFECT_NAMES_TR = {
    "Çatlak",
    "Yıldız çatlağı",
    "Parçalanma",
    "Kısa devre",
}
