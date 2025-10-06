from __future__ import annotations
import json, os
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
                    "conf_th": float(data.get("conf_th", 0.60)),
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