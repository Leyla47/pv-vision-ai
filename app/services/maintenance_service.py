"""Panel sağlık ve kusur sonuçlarından tekrarsız bakım önerileri üretir."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.health_service import HealthAssessment
from app.services.quality_service import QualityAssessment


RISK_RECOMMENDATIONS = {
    "Düşük": (
        "Panel genel olarak sağlıklı görünmektedir.",
        "Rutin görsel kontrol ve planlı bakım yeterlidir.",
    ),
    "Orta": (
        "Kusurlu bölgelerin detaylı kontrol edilmesi önerilir.",
        "IV eğrisi veya elektriksel performans testi yapılması önerilir.",
        "Panel kısa vadede yeniden görüntülenmelidir.",
    ),
    "Yüksek": (
        "Panel detaylı teknik incelemeye alınmalıdır.",
        "Elektriksel güvenlik doğrulanana kadar panelin kullanım koşulları değerlendirilmelidir.",
        "Gerekirse panel kullanım dışı bırakılmalı veya değiştirilmelidir.",
    ),
}

CRACK_RECOMMENDATION = (
    "Panel üzerinde çatlak tespit edilmiştir. Kusurun ilerleyip ilerlemediğinin "
    "kontrol edilmesi ve panelin elektriksel performansının ölçülmesi önerilir."
)
MECHANICAL_RECOMMENDATION = (
    "Panelin mekanik bütünlüğü, kenar hasarı ve yalıtım güvenliği kontrol edilmelidir."
)
INTERCONNECTION_RECOMMENDATION = (
    "Hücre bağlantıları ve akım yolları EL karşılaştırması ve elektriksel test ile kontrol edilmelidir."
)

CLASS_RECOMMENDATIONS = {
    "Çatlak": CRACK_RECOMMENDATION,
    "Yıldız çatlağı": CRACK_RECOMMENDATION,
    "Kısa devre": (
        "Kısa devre şüphesi nedeniyle panel güvenli biçimde izole edilmeli ve yetkili uzman "
        "tarafından elektriksel ve termal incelemeye alınmalıdır."
    ),
    "Parçalanma": MECHANICAL_RECOMMENDATION,
    "Köşe kusuru": MECHANICAL_RECOMMENDATION,
    "Siyah çekirdek": (
        "Siyah çekirdek bölgesi için sıcak nokta, akım uyumsuzluğu ve IV performansı kontrol edilmelidir."
    ),
    "Yatay hizasızlık": INTERCONNECTION_RECOMMENDATION,
    "Dikey hizasızlık": INTERCONNECTION_RECOMMENDATION,
    "Kalın çizgi": INTERCONNECTION_RECOMMENDATION,
    "Baskı hatası": INTERCONNECTION_RECOMMENDATION,
    "Çizik": (
        "Çizik bölgesinde yüzey kaplaması, yalıtım ve nem girişi riski görsel olarak kontrol edilmelidir."
    ),
    "Parmak izi kusuru": (
        "Yüzey üretici talimatına uygun temizlenmeli ve kalıcı kusuru ayırmak için EL görüntüsü tekrarlanmalıdır."
    ),
}


@dataclass(frozen=True)
class MaintenancePlan:
    """Risk seviyesi ve öncelikli bakım önerileri."""

    risk_level: str
    recommendations: tuple[str, ...]
    considered_classes: tuple[str, ...]


def build_maintenance_plan(
    assessment: QualityAssessment,
    health: HealthAssessment,
) -> MaintenancePlan:
    """Risk önerilerini en etkili üç kusur sınıfının önerileriyle birleştirir."""
    detected_classes = set(assessment.detailed_detections["Kusur sınıfı"].astype(str))
    considered = tuple(
        impact.label
        for impact in health.class_impacts
        if impact.label in detected_classes
    )[:3]
    recommendations = list(RISK_RECOMMENDATIONS[health.risk_level])
    for class_name in considered:
        recommendation = CLASS_RECOMMENDATIONS.get(class_name)
        if recommendation and recommendation not in recommendations:
            recommendations.append(recommendation)
    return MaintenancePlan(
        risk_level=health.risk_level,
        recommendations=tuple(recommendations),
        considered_classes=considered,
    )
