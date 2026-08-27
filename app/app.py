"""PV Vision AI Streamlit uygulamasının giriş noktası."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.health_service import HealthAssessment, assess_panel_health  # noqa: E402
from app.services.maintenance_service import (  # noqa: E402
    MaintenancePlan,
    build_maintenance_plan,
)
from app.services.model_service import DetectionResult, load_model, predict_image  # noqa: E402
from app.services.performance_service import (  # noqa: E402
    PerformanceEstimate,
    estimate_production_performance,
)
from app.services.quality_service import QualityAssessment, assess_quality  # noqa: E402
from app.utils.image_utils import bgr_image_to_png_bytes, open_uploaded_image  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_REFERENCE_PRICE,
    DEFAULT_REFERENCE_POWER_W,
    MAX_REFERENCE_POWER_W,
    MIN_REFERENCE_PRICE,
    MIN_REFERENCE_POWER_W,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    REFERENCE_PRICE_STEP,
    REPORTS_DIR,
    SUPPORTED_PRICE_CURRENCIES,
)
from model_registry import load_model_metadata  # noqa: E402


MAX_UPLOAD_SIZE = 20 * 1024 * 1024
MIN_IMAGE_DIMENSION = 128
ANALYSIS_RESULT_KEY = "pv_vision_analysis_result"
ANALYSIS_REQUEST_KEY = "pv_vision_analysis_request"
HERO_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "solar-panel-hero.webp"
ANALYSIS_DISCLAIMER = (
    "Sağlık skoru, üretim performansı ve ekonomik kayıp değerleri görüntü üzerinden "
    "tespit edilen kusurlara dayalı tahmini sonuçlardır. Gerçek elektriksel ölçüm, "
    "laboratuvar testi veya kesin finansal değer yerine geçmez."
)


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path: str, model_version: int) -> object:
    """Ağırlık dosyası değişene kadar YOLO modelini Streamlit önbelleğinde tutar."""
    del model_version
    return load_model(Path(model_path))


def main() -> None:
    """Streamlit uygulamasını çalıştırır."""
    st.set_page_config(page_title="PV Vision AI", page_icon="☀", layout="wide")
    model_view = _get_model_view()
    _inject_styles()
    _render_navigation(model_view)
    _render_hero(model_view)

    st.markdown('<div id="analiz" class="pv-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pv-section-heading">
            <span class="pv-eyebrow">EL GÖRÜNTÜ ANALİZİ</span>
            <h2>Hücre görüntüsünü incele</h2>
            <p>Görüntüyü yükle, güven eşiğini belirle ve kusur bölgelerini modelle işaretle.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_columns = st.columns([2.15, 1], gap="large")
    with control_columns[0]:
        uploaded_file = st.file_uploader(
            "EL görüntüsünü buraya sürükle veya seç",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            help="En fazla 20 MB boyutunda bir elektrolüminesans görüntüsü seç.",
        )
    with control_columns[1]:
        confidence = st.slider(
            "Güven eşiği",
            min_value=0.05,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Düşük değer daha fazla, yüksek değer daha seçici tespit üretir.",
        )
        reference_power_w = float(
            st.number_input(
                "Referans panel gücü (W)",
                min_value=MIN_REFERENCE_POWER_W,
                max_value=MAX_REFERENCE_POWER_W,
                value=DEFAULT_REFERENCE_POWER_W,
                step=10,
                help="Panelin nominal etiket gücünü Watt cinsinden gir.",
            )
        )
        price_enabled = st.toggle(
            "Fiyat önerisini hesapla",
            value=False,
            help="Referans fiyat girilirse kalite sınıfına göre tahmini fiyat hesaplanır.",
        )
        reference_price: float | None = None
        currency = "TRY"
        if price_enabled:
            reference_price = float(
                st.number_input(
                    "Referans panel fiyatı",
                    min_value=MIN_REFERENCE_PRICE,
                    value=DEFAULT_REFERENCE_PRICE,
                    step=REFERENCE_PRICE_STEP,
                    format="%.2f",
                )
            )
            currency = st.selectbox(
                "Para birimi",
                options=SUPPORTED_PRICE_CURRENCIES,
                index=0,
            )
        model_available = MODEL_WEIGHTS_PATH.exists()
        analyze_clicked = st.button(
            "Analiz Et",
            type="primary",
            disabled=uploaded_file is None or not model_available,
            use_container_width=True,
        )

    if not model_available:
        st.error("Model dosyası bulunamadı. Önce bir best.pt modeli üretilmelidir.")
        _render_details(model_view)
        return
    if uploaded_file is None:
        _render_empty_state()
        _render_details(model_view)
        return

    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) > MAX_UPLOAD_SIZE:
        st.error("Görüntü dosyası 20 MB sınırını aşıyor.")
        _render_details(model_view)
        return

    try:
        image = open_uploaded_image(file_bytes)
    except ValueError as exc:
        st.error(str(exc))
        _render_details(model_view)
        return
    if min(image.size) < MIN_IMAGE_DIMENSION:
        st.error("Görüntünün genişliği ve yüksekliği en az 128 piksel olmalıdır.")
        _render_details(model_view)
        return

    model_version = MODEL_WEIGHTS_PATH.stat().st_mtime_ns
    image_digest = hashlib.sha256(file_bytes).hexdigest()
    request_key = f"{image_digest}:{confidence:.2f}:{model_version}"

    if analyze_clicked:
        try:
            with st.spinner("Model görüntüyü analiz ediyor..."):
                model = get_cached_model(str(MODEL_WEIGHTS_PATH), model_version)
                result = predict_image(model, image, confidence=confidence)
        except FileNotFoundError:
            st.error("Model dosyası yüklenemedi. Eğitim çıktısındaki best.pt dosyasını kontrol et.")
            _render_details(model_view)
            return
        except Exception as exc:
            st.error(f"Analiz sırasında beklenmeyen bir hata oluştu: {exc}")
            _render_details(model_view)
            return

        st.session_state[ANALYSIS_REQUEST_KEY] = request_key
        st.session_state[ANALYSIS_RESULT_KEY] = result

    result = _current_result(request_key)
    _render_images(image, result)
    if result is None:
        st.info("Görüntü hazır. Analizi başlatmak için Analiz Et düğmesine bas.")
        _render_details(model_view)
        return

    _render_analysis(
        result,
        uploaded_file.name,
        image.size,
        reference_power_w=reference_power_w,
        reference_price=reference_price,
        currency=currency,
    )
    _render_details(model_view)


def _get_model_view() -> dict[str, object]:
    """Model metadatasını arayüzün tek bir görünüm modeline dönüştürür."""
    try:
        metadata = load_model_metadata(MODEL_METADATA_PATH)
    except ValueError as exc:
        return _unavailable_model_view(str(exc), "Kayıt hatası")
    if metadata is None:
        return _unavailable_model_view(
            "Model mevcut, ancak eğitim kimlik bilgisi bulunamadı.",
            "Model bekleniyor",
        )

    stage = str(metadata.get("stage") or "unknown")
    completed = metadata.get("completed_epochs") or 0
    target = metadata.get("target_epochs") or "?"
    best_epoch = metadata.get("best_epoch") or "?"
    training_metrics = metadata.get("metrics") or {}
    evaluation = metadata.get("evaluation") or {}
    if stage == "final" and evaluation.get("test"):
        metrics = evaluation["test"]
        metric_source = "test"
    elif evaluation.get("validation"):
        metrics = evaluation["validation"]
        metric_source = "validation"
    else:
        metrics = training_metrics
        metric_source = "eğitim sırasındaki validation"

    stage_content = {
        "final": (
            "Final model",
            "success",
            f"Final model hazır. En iyi sonuç {best_epoch}. epoch'tan; eğitim {completed}/{target} epoch.",
        ),
        "candidate": (
            "Aday model",
            "info",
            f"Eğitim tamamlandı. {best_epoch}. epoch'taki model final değerlendirmesini bekliyor.",
        ),
        "interim": (
            "Ara model",
            "warning",
            f"Şu anda {best_epoch}. epoch'taki en iyi ara model kullanılıyor. Eğitim {completed}/{target} epoch.",
        ),
        "smoke": (
            "Deneme modeli",
            "warning",
            "Bu model yalnızca eğitim ve uygulama hattını doğrulamak için üretildi.",
        ),
    }
    stage_label, tone, message = stage_content.get(
        stage,
        ("Model durumu", "warning", "Model aşaması doğrulanamadı."),
    )
    return {
        "stage": stage,
        "stage_label": stage_label,
        "tone": tone,
        "message": message,
        "completed": completed,
        "target": target,
        "best_epoch": best_epoch,
        "map50": _format_metric(metrics.get("map50")),
        "map50_95": _format_metric(metrics.get("map50_95")),
        "metric_source": metric_source,
    }


def _unavailable_model_view(message: str, label: str) -> dict[str, object]:
    return {
        "stage": "unknown",
        "stage_label": label,
        "tone": "warning",
        "message": message,
        "completed": 0,
        "target": "?",
        "best_epoch": "?",
        "map50": "-",
        "map50_95": "-",
        "metric_source": "kullanılamıyor",
    }


def _render_navigation(model_view: dict[str, object]) -> None:
    tone = _safe_css_token(model_view["tone"])
    st.markdown(
        f"""
        <nav class="pv-nav" aria-label="Ana navigasyon">
            <a class="pv-brand" href="#pv-hero" aria-label="PV Vision AI ana bölüm">
                <span class="pv-brand-mark" aria-hidden="true"></span>
                <span>PV Vision <strong>AI</strong></span>
            </a>
            <div class="pv-nav-links">
                <a href="#analiz">Analiz</a><a href="#model">Model</a><a href="#kapsam">Kapsam</a>
            </div>
            <span class="pv-status pv-status-{tone}"><span class="pv-status-dot"></span>{model_view['stage_label']}</span>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(model_view: dict[str, object]) -> None:
    st.markdown(
        f"""
        <section id="pv-hero" class="pv-hero">
            <div class="pv-hero-copy">
                <span class="pv-eyebrow"><span class="pv-live-dot"></span> FOTOVOLTAİK GÖRÜNTÜ İŞLEME</span>
                <h1>Güneş Hücrelerinde <span>Yapay Zekâ</span> Destekli Kusur Tespiti</h1>
                <p>Elektrolüminesans görüntülerini analiz edin, kusur bölgelerini görün ve sonuçları raporlayın.</p>
                <a class="pv-hero-action" href="#analiz">Analize başla <span aria-hidden="true">↓</span></a>
            </div>
            <div class="pv-metric-grid" aria-label="Model özeti">
                {_metric_card("12", "Kusur sınıfı")}
                {_metric_card(f"{model_view['completed']}/{model_view['target']}", "Tamamlanan epoch")}
                {_metric_card(str(model_view["map50"]), "mAP50")}
                {_metric_card(str(model_view["stage_label"]), "Model aşaması")}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(value: str, label: str) -> str:
    return f'<div class="pv-metric"><strong>{value}</strong><span>{label}</span></div>'


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="pv-empty-state">
            <span class="pv-empty-icon" aria-hidden="true">＋</span>
            <strong>Analiz alanı hazır</strong>
            <p>Desteklenen bir EL görüntüsü yüklediğinizde önizleme ve model sonucu burada görünecek.</p>
            <span>JPG, PNG, BMP veya TIFF · En fazla 20 MB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_details(model_view: dict[str, object]) -> None:
    st.markdown('<div id="model" class="pv-anchor"></div>', unsafe_allow_html=True)
    st.markdown('<div class="pv-detail-title">Model ve kapsam bilgileri</div>', unsafe_allow_html=True)
    with st.expander("Model durumu ve metrik kaynağı"):
        tone = str(model_view["tone"])
        message = str(model_view["message"])
        if tone == "success":
            st.success(message)
        elif tone == "info":
            st.info(message)
        else:
            st.warning(message)

        metric_columns = st.columns(4)
        metric_columns[0].metric("Tamamlanan epoch", f"{model_view['completed']}/{model_view['target']}")
        metric_columns[1].metric("En iyi epoch", str(model_view["best_epoch"]))
        metric_columns[2].metric("mAP50", str(model_view["map50"]))
        metric_columns[3].metric("mAP50-95", str(model_view["map50_95"]))
        st.caption(f"Gösterilen model metriği kaynağı: {model_view['metric_source']}.")

    st.markdown('<div id="kapsam" class="pv-anchor"></div>', unsafe_allow_html=True)
    _render_dataset_coverage()


def _render_dataset_coverage() -> None:
    analysis_path = REPORTS_DIR / "dataset_class_distribution.json"
    if not analysis_path.exists():
        return
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    limited_classes = [
        item
        for item in payload.get("classes", [])
        if item.get("training_support") != "Yeterli"
        or item.get("validation_coverage") != "Kapsanıyor"
    ]
    if not limited_classes:
        return

    with st.expander("Model kapsamı ve veri sınırlılıkları"):
        st.warning(
            "Bazı kusur sınıfları veri setinde çok az örneğe sahip. "
            "Bu sınıflardaki tahminler daha düşük güvenilirlikte olabilir."
        )
        rows = [
            {
                "Kusur sınıfı": item["class_name"],
                "Eğitim kutusu": item["train_objects"],
                "Validation kutusu": item["val_objects"],
                "Eğitim desteği": item["training_support"],
                "Validation kapsaması": item["validation_coverage"],
            }
            for item in limited_classes
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _current_result(request_key: str) -> DetectionResult | None:
    if st.session_state.get(ANALYSIS_REQUEST_KEY) != request_key:
        return None
    result = st.session_state.get(ANALYSIS_RESULT_KEY)
    return result if isinstance(result, DetectionResult) else None


def _render_images(image: object, result: DetectionResult | None) -> None:
    st.markdown('<div class="pv-result-heading">Görüntü karşılaştırması</div>', unsafe_allow_html=True)
    left_column, right_column = st.columns(2, gap="large")
    with left_column:
        st.markdown("#### Orijinal görüntü")
        st.image(image, use_column_width=True)
    with right_column:
        st.markdown("#### İşaretlenmiş sonuç")
        if result is None:
            st.markdown(
                '<div class="pv-image-placeholder"><strong>Sonuç bekleniyor</strong>'
                '<span>Analiz tamamlandığında tespit kutuları burada gösterilecek.</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.image(result.annotated_image, channels="BGR", use_column_width=True)


def _render_analysis(
    result: DetectionResult,
    original_filename: str,
    image_size: tuple[int, int],
    *,
    reference_power_w: float,
    reference_price: float | None,
    currency: str,
) -> None:
    assessment = assess_quality(
        result.detections,
        image_size,
        reference_price=reference_price,
        currency=currency,
    )
    performance = estimate_production_performance(assessment, reference_power_w)
    health = assess_panel_health(assessment)
    maintenance = build_maintenance_plan(assessment, health)
    st.markdown('<div class="pv-result-heading">Analiz özeti</div>', unsafe_allow_html=True)
    if result.detections.empty:
        st.info(result.summary)
    else:
        st.success(result.summary)

    total_detections = len(result.detections)
    metric_columns = st.columns(4)
    metric_columns[0].metric("Toplam kusur", total_detections)
    metric_columns[1].metric(
        "Kusurlu alan",
        f"%{assessment.covered_area_percent:.2f}",
    )
    metric_columns[2].metric("Panel Sağlık Skoru", f"{health.score:.1f}/100")
    metric_columns[3].metric(
        "Genel Sağlık Durumu",
        health.status,
    )
    quality_columns = st.columns(3)
    quality_columns[0].metric("Kalite puanı", f"{assessment.score:.1f}/100")
    quality_columns[1].metric(
        "Kalite sınıfı",
        f"{assessment.grade} · {assessment.grade_label}",
    )
    quality_columns[2].metric("Risk Seviyesi", health.risk_level)
    st.progress(
        int(round(health.score)),
        text=f"Panel sağlık skoru: {health.score:.1f}/100",
    )
    st.caption(
        "Alan oranı, YOLO tespit kutularının yüklenen görüntüde kapladığı "
        "çakışmasız yaklaşık alandır; fiziksel panel ölçümü değildir."
    )

    st.markdown("#### Tahmini üretim performansı")
    performance_columns = st.columns(4)
    performance_columns[0].metric(
        "Tahmini Üretim Performansı",
        f"%{performance.performance_percent:.1f}",
    )
    performance_columns[1].metric(
        "Tahmini Performans Kaybı",
        f"%{performance.performance_loss_percent:.1f}",
    )
    performance_columns[2].metric(
        "Referans Panel Gücü",
        f"{performance.reference_power_w:.1f} W",
    )
    performance_columns[3].metric(
        "Tahmini Panel Gücü",
        f"{performance.estimated_power_w:.1f} W",
    )

    if assessment.suggested_price is not None:
        st.markdown("#### Tahmini ekonomik etki")
        price_columns = st.columns(4)
        price_columns[0].metric(
            "Referans Panel Fiyatı",
            _format_price(assessment.reference_price, assessment.currency),
        )
        price_columns[1].metric(
            "Tahmini Değer Kaybı",
            f"%{assessment.value_loss_percent:.1f}",
        )
        price_columns[2].metric(
            "Tahmini Kayıp Tutarı",
            _format_price(assessment.value_loss_amount, assessment.currency),
        )
        price_columns[3].metric(
            "Tahmini Panel Değeri",
            _format_price(assessment.suggested_price, assessment.currency),
        )

    st.markdown("#### Sağlık skorunun açıklaması")
    st.dataframe(
        [
            {"Etki": impact.label, "Puan düşüşü": f"-{impact.points:.1f}"}
            for impact in health.impacts
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Bakım ve kontrol önerileri")
    st.markdown("\n".join(f"- {item}" for item in maintenance.recommendations))
    st.warning(ANALYSIS_DISCLAIMER)

    st.markdown("#### Tespit detayları")
    if result.detections.empty:
        st.write("Seçilen güven eşiğinde gösterilecek tespit yok.")
    else:
        st.dataframe(
            assessment.detailed_detections,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Güven (%)": st.column_config.ProgressColumn(
                    "Güven (%)", min_value=0.0, max_value=100.0, format="%.1f%%"
                )
            },
        )
        st.markdown("#### Sınıf özeti")
        st.dataframe(assessment.class_summary, use_container_width=True, hide_index=True)
        chart_columns = st.columns(2, gap="large")
        with chart_columns[0]:
            st.markdown("##### Kusur adedi")
            st.bar_chart(
                assessment.class_summary,
                x="Kusur sınıfı",
                y="Tespit sayısı",
                color="#F5C542",
                horizontal=True,
                height=260,
            )
        with chart_columns[1]:
            st.markdown("##### Görüntüde kaplanan alan")
            st.bar_chart(
                assessment.class_summary,
                x="Kusur sınıfı",
                y="Birleşik alan (%)",
                color="#FFD76A",
                horizontal=True,
                height=260,
            )

    stem = Path(original_filename).stem
    csv_bytes = assessment.detailed_detections.to_csv(index=False).encode("utf-8-sig")
    report_bytes = _build_text_report(
        result,
        assessment,
        performance,
        health,
        maintenance,
    ).encode("utf-8")
    st.markdown("#### Çıktıları indir")
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "İşaretli görüntü",
        data=bgr_image_to_png_bytes(result.annotated_image),
        file_name=f"{stem}_pv_vision.png",
        mime="image/png",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Tespit tablosu",
        data=csv_bytes,
        file_name=f"{stem}_tespitler.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[2].download_button(
        "Analiz özeti",
        data=report_bytes,
        file_name=f"{stem}_analiz.txt",
        mime="text/plain",
        use_container_width=True,
    )


def _build_text_report(
    result: DetectionResult,
    assessment: QualityAssessment,
    performance: PerformanceEstimate,
    health: HealthAssessment,
    maintenance: MaintenancePlan,
) -> str:
    lines = [
        "PV Vision AI Analiz Özeti",
        "--------------------------",
        result.summary,
        f"Toplam kusur sayısı: {len(result.detections)}",
        f"Kusur kutularının birleşik alanı: %{assessment.covered_area_percent:.3f}",
        f"Kalite puanı: {assessment.score:.1f}/100",
        f"Kalite sınıfı: {assessment.grade} - {assessment.grade_label}",
        f"Alan cezası: {assessment.area_penalty:.3f}",
        f"Adet cezası: {assessment.count_penalty:.3f}",
        f"Panel sağlık skoru: {health.score:.1f}/100",
        f"Genel sağlık durumu: {health.status}",
        f"Risk seviyesi: {health.risk_level}",
        f"Tahmini üretim performansı: %{performance.performance_percent:.1f}",
        f"Tahmini performans kaybı: %{performance.performance_loss_percent:.1f}",
        f"Referans panel gücü: {performance.reference_power_w:.1f} W",
        f"Tahmini panel gücü: {performance.estimated_power_w:.1f} W",
        "Sağlık skoru etkileri:",
    ]
    lines.extend(f"- {impact.label}: -{impact.points:.3f} puan" for impact in health.impacts)
    lines.append("Bakım ve kontrol önerileri:")
    lines.extend(f"- {item}" for item in maintenance.recommendations)
    if assessment.suggested_price is not None:
        lines.extend(
            [
                f"Referans fiyat: {_format_price(assessment.reference_price, assessment.currency)}",
                f"Kalite katsayısı: {assessment.price_coefficient:.2f}",
                f"Tahmini değer kaybı: %{assessment.value_loss_percent:.1f}",
                f"Tahmini kayıp tutarı: {_format_price(assessment.value_loss_amount, assessment.currency)}",
                f"Tahmini panel değeri: {_format_price(assessment.suggested_price, assessment.currency)}",
            ]
        )
    lines.append(f"Not: {ANALYSIS_DISCLAIMER}")
    if not result.detections.empty:
        counts = result.detections["Kusur sınıfı"].value_counts()
        lines.extend(f"- {class_name}: {count}" for class_name, count in counts.items())
    return "\n".join(lines) + "\n"


def _format_price(value: float | None, currency: str) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f} {currency}"


def _format_metric(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _safe_css_token(value: object) -> str:
    token = str(value)
    return token if token in {"success", "info", "warning"} else "warning"


@lru_cache(maxsize=1)
def _hero_data_uri() -> str:
    if not HERO_IMAGE_PATH.exists():
        return ""
    encoded = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def _inject_styles() -> None:
    hero_uri = _hero_data_uri()
    st.markdown(
        f"""
        <style>
        :root {{ --pv-bg:#090A0B; --pv-surface:#141518; --pv-border:#24262B; --pv-yellow:#F5C542; --pv-yellow-soft:#FFD76A; --pv-text:#F7F8FA; --pv-muted:#A7ABB3; }}
        html {{ scroll-behavior:smooth; }}
        body {{ background:var(--pv-bg); }}
        [data-testid="stAppViewContainer"] {{ background:var(--pv-bg); color:var(--pv-text); }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{ display:none; }}
        [data-testid="stMainBlockContainer"] {{ max-width:1240px; padding:84px 32px 72px; }}
        .pv-anchor {{ position:relative; top:-88px; visibility:hidden; }}
        .pv-nav {{ position:fixed; z-index:999; top:0; left:0; right:0; height:68px; display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:28px; padding:0 max(28px, calc((100vw - 1176px) / 2)); background:rgba(9,10,11,.92); border-bottom:1px solid var(--pv-border); backdrop-filter:blur(16px); }}
        .pv-brand {{ display:inline-flex; align-items:center; gap:11px; width:max-content; color:var(--pv-text)!important; font-size:20px; font-weight:720; text-decoration:none!important; white-space:nowrap; }}
        .pv-brand strong {{ color:var(--pv-yellow); }}
        .pv-brand-mark {{ width:24px; height:24px; border:2px solid var(--pv-yellow); background:linear-gradient(90deg,transparent 46%,var(--pv-yellow) 47%,var(--pv-yellow) 53%,transparent 54%),linear-gradient(0deg,transparent 46%,var(--pv-yellow) 47%,var(--pv-yellow) 53%,transparent 54%); transform:rotate(45deg); box-shadow:0 0 18px rgba(245,197,66,.18); }}
        .pv-nav-links {{ display:flex; gap:34px; align-items:center; }}
        .pv-nav-links a {{ color:#D9DBDF!important; font-size:14px; font-weight:600; text-decoration:none!important; }}
        .pv-nav-links a:hover {{ color:var(--pv-yellow)!important; }}
        .pv-status {{ justify-self:end; display:inline-flex; align-items:center; gap:8px; min-height:34px; padding:0 13px; border:1px solid #34363B; border-radius:6px; background:#17181B; color:#E8E9EC; font-size:12px; font-weight:650; white-space:nowrap; }}
        .pv-status-dot {{ width:7px; height:7px; border-radius:50%; background:var(--pv-yellow); }}
        .pv-status-success .pv-status-dot {{ background:#69D391; }} .pv-status-info .pv-status-dot {{ background:#71B7FF; }}
        .pv-hero {{ position:relative; min-height:500px; display:flex; flex-direction:column; justify-content:space-between; overflow:hidden; border-bottom:1px solid #373024; background-image:linear-gradient(90deg,rgba(5,6,7,.98) 0%,rgba(5,6,7,.88) 42%,rgba(5,6,7,.36) 77%,rgba(5,6,7,.22) 100%),linear-gradient(0deg,rgba(9,10,11,.96) 0%,rgba(9,10,11,.04) 50%),url('{hero_uri}'); background-position:center; background-size:cover; padding:56px 52px 28px; }}
        .pv-hero::after {{ content:""; position:absolute; inset:0; pointer-events:none; border:1px solid rgba(255,255,255,.06); }}
        .pv-hero-copy {{ position:relative; z-index:1; max-width:790px; }}
        .pv-eyebrow {{ display:inline-flex; align-items:center; gap:9px; color:var(--pv-yellow-soft); font-size:12px; font-weight:750; letter-spacing:0; }}
        .pv-live-dot {{ width:8px; height:8px; border-radius:50%; background:var(--pv-yellow); box-shadow:0 0 14px rgba(245,197,66,.7); }}
        .pv-hero h1 {{ max-width:760px; margin:20px 0 18px; color:#FFF; font-size:53px; line-height:1.06; font-weight:760; letter-spacing:0; }}
        .pv-hero h1 span {{ color:var(--pv-yellow); }}
        .pv-hero-copy>p {{ max-width:650px; margin:0 0 24px; color:#C6C8CE; font-size:17px; line-height:1.55; }}
        .pv-hero-action {{ display:inline-flex; align-items:center; gap:12px; min-height:42px; padding:0 17px; border:1px solid rgba(245,197,66,.58); border-radius:6px; background:rgba(9,10,11,.48); color:var(--pv-yellow-soft)!important; font-size:14px; font-weight:700; text-decoration:none!important; }}
        .pv-metric-grid {{ position:relative; z-index:1; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin-top:34px; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.12); }}
        .pv-metric {{ min-width:0; min-height:76px; display:flex; flex-direction:column; justify-content:center; padding:13px 18px; background:rgba(13,14,16,.82); backdrop-filter:blur(9px); }}
        .pv-metric strong {{ overflow-wrap:anywhere; color:#FFF; font-size:22px; line-height:1.1; font-weight:760; }} .pv-metric span {{ margin-top:7px; color:#999DA6; font-size:11px; font-weight:650; }}
        .pv-section-heading {{ margin:58px 0 24px; max-width:720px; }} .pv-section-heading h2 {{ margin:8px 0 7px; color:#F7F8FA; font-size:30px; line-height:1.15; }} .pv-section-heading p {{ margin:0; color:var(--pv-muted); font-size:15px; }}
        [data-testid="stFileUploader"] {{ min-height:184px; padding:18px; border:1px solid var(--pv-border); border-radius:8px; background:var(--pv-surface); }}
        [data-testid="stFileUploaderDropzone"] {{ min-height:132px; border:1px dashed #55504A; border-radius:6px; background:#101113; }}
        [data-testid="stFileUploaderDropzone"]:hover {{ border-color:var(--pv-yellow); background:#15140F; }} [data-testid="stFileUploader"] small {{ color:#898D95!important; }}
        [data-testid="stFileUploaderDropzoneInstructions"]>div>span {{ font-size:0; }}
        [data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {{ content:"Görüntüyü sürükleyip bırakın"; font-size:14px; }}
        [data-testid="stFileUploaderDropzoneInstructions"] small {{ font-size:0; }}
        [data-testid="stFileUploaderDropzoneInstructions"] small::after {{ content:"Dosya başına en fazla 20 MB"; font-size:12px; }}
        [data-testid="stFileUploaderDropzone"]>[data-testid="stBaseButton-secondary"] {{ font-size:0; }}
        [data-testid="stFileUploaderDropzone"]>[data-testid="stBaseButton-secondary"]::after {{ content:"Dosya seç"; font-size:14px; }}
        [data-testid="stSlider"] {{ padding:18px; border:1px solid var(--pv-border); border-radius:8px; background:var(--pv-surface); }} [data-baseweb="slider"] [role="slider"] {{ background:var(--pv-yellow)!important; }}
        [data-testid="stProgress"]>div>div>div>div {{ background-color:var(--pv-yellow)!important; }}
        .stButton>button[kind="primary"] {{ min-height:48px; margin-top:12px; border:1px solid var(--pv-yellow); border-radius:6px; background:var(--pv-yellow); color:#111214; font-weight:760; }}
        .stButton>button[kind="primary"]:hover {{ border-color:var(--pv-yellow-soft); background:var(--pv-yellow-soft); color:#111214; }} .stButton>button:disabled {{ opacity:.44; }}
        .pv-empty-state {{ min-height:220px; display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:28px; padding:32px; border:1px solid var(--pv-border); border-radius:8px; background:#101113; text-align:center; }}
        .pv-empty-icon {{ width:38px; height:38px; display:grid; place-items:center; margin-bottom:14px; border:1px solid #4D4738; color:var(--pv-yellow); font-size:24px; }}
        .pv-empty-state strong {{ color:#F5F6F8; font-size:17px; }} .pv-empty-state p {{ max-width:520px; margin:8px 0 10px; color:var(--pv-muted); }} .pv-empty-state>span:last-child {{ color:#71757E; font-size:12px; }}
        .pv-result-heading,.pv-detail-title {{ margin:46px 0 18px; color:#F6F7F9; font-size:22px; font-weight:730; }}
        [data-testid="stImage"] img {{ border:1px solid var(--pv-border); border-radius:6px; }}
        .pv-image-placeholder {{ min-height:280px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:8px; padding:24px; border:1px dashed #3B3D42; border-radius:6px; background:#101113; color:var(--pv-muted); text-align:center; }} .pv-image-placeholder strong {{ color:#DCDDDF; }}
        [data-testid="stMetric"] {{ min-height:96px; padding:16px; border:1px solid var(--pv-border); border-radius:6px; background:var(--pv-surface); }} [data-testid="stMetricValue"] {{ color:var(--pv-yellow-soft); }}
        [data-testid="stMetricLabel"] p,[data-testid="stMetricValue"]>div {{ white-space:normal!important; overflow:visible!important; text-overflow:clip!important; line-height:1.15!important; }}
        [data-testid="stAlert"] {{ border-radius:6px; }} [data-testid="stExpander"] {{ border:1px solid var(--pv-border); border-radius:6px; background:var(--pv-surface); }}
        [data-testid="stDownloadButton"] button {{ min-height:44px; border:1px solid #3A3C42; border-radius:6px; background:#17181B; color:#F2F3F5; }} [data-testid="stDownloadButton"] button:hover {{ border-color:var(--pv-yellow); color:var(--pv-yellow-soft); }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--pv-border); border-radius:6px; overflow:hidden; }} hr {{ border-color:var(--pv-border)!important; }}
        @media (max-width:760px) {{
            [data-testid="stMainBlockContainer"] {{ padding:72px 16px 48px; }}
            .pv-nav {{ height:58px; grid-template-columns:1fr auto; padding:0 16px; }} .pv-nav-links {{ display:none; }} .pv-brand {{ font-size:17px; }} .pv-brand-mark {{ width:20px; height:20px; }} .pv-status {{ min-height:30px; padding:0 9px; font-size:10px; }}
            .pv-hero {{ min-height:548px; padding:38px 20px 20px; background-position:61% center; }} .pv-hero h1 {{ margin-top:17px; font-size:35px; line-height:1.08; }} .pv-hero-copy>p {{ font-size:15px; }}
            .pv-metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .pv-metric {{ min-height:68px; padding:11px 12px; }} .pv-metric strong {{ font-size:18px; }}
            .pv-section-heading {{ margin-top:42px; }} .pv-section-heading h2 {{ font-size:25px; }}
            [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
            [data-testid="stHorizontalBlock"]>[data-testid="column"] {{ flex:1 1 100%!important; width:100%!important; min-width:0!important; }}
            [data-testid="stHorizontalBlock"]:has(>[data-testid="column"]:nth-child(4))>[data-testid="column"] {{ flex:0 0 calc(50% - .5rem)!important; width:calc(50% - .5rem)!important; }}
            [data-testid="stMetric"] {{ min-height:108px; }}
            [data-testid="stMetricValue"] {{ font-size:24px; }}
            .pv-image-placeholder {{ min-height:210px; }} .pv-result-heading,.pv-detail-title {{ margin-top:36px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
