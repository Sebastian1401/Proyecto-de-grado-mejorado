# app/services/settings_service.py
from __future__ import annotations
import json, os, threading
from dataclasses import dataclass, asdict

SETTINGS_FILE = os.environ.get("THRESHOLDS_JSON", "thresholds.json")

@dataclass
class Thresholds:
    conf_th: float = 0.30
    iou_th:  float = 0.50
    min_box_frac: float = 0.003

class SettingsService:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or SETTINGS_FILE

    def load(self) -> Thresholds:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return Thresholds(**{
                    "conf_th": float(data.get("conf_th", 0.50)),
                    "iou_th": float(data.get("iou_th", 0.30)),
                    "min_box_frac": float(data.get("min_box_frac", 0.003)),
                })
            except Exception:
                pass
        return Thresholds()

    def save(self, t: Thresholds) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(t), f, indent=2)

    def reset(self) -> Thresholds:
        t = Thresholds()
        self.save(t)
        return t


# === Cache en memoria para thresholds (single process) ===
class _ThresholdsCache:
    def __init__(self, path: str) -> None:
        self._lock = threading.RLock()
        self._path = path
        self._svc = SettingsService(path)
        # estado
        self._t: Thresholds = self._svc.load()
        self._version: int = 1

    def snapshot(self) -> tuple[Thresholds, int]:
        """Devuelve copia inmutable + versión actual (para depurar si quieres)."""
        with self._lock:
            # devolvemos un dataclass nuevo para evitar mutaciones externas
            t = Thresholds(self._t.conf_th, self._t.iou_th, self._t.min_box_frac)
            return t, self._version

    def update(self, **kwargs) -> Thresholds:
        """Actualiza campos, persiste y aumenta versión."""
        with self._lock:
            if "conf_th" in kwargs:      self._t.conf_th = float(kwargs["conf_th"])
            if "iou_th" in kwargs:       self._t.iou_th = float(kwargs["iou_th"])
            if "min_box_frac" in kwargs: self._t.min_box_frac = float(kwargs["min_box_frac"])
            self._svc.save(self._t)
            self._version += 1
            return Thresholds(self._t.conf_th, self._t.iou_th, self._t.min_box_frac)

    def reset(self) -> Thresholds:
        with self._lock:
            self._t = self._svc.reset()
            self._version += 1
            return Thresholds(self._t.conf_th, self._t.iou_th, self._t.min_box_frac)

# instancia única de cache
THRESHOLDS_CACHE = _ThresholdsCache(SETTINGS_FILE)
