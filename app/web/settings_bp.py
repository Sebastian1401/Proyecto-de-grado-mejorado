from flask import Blueprint, jsonify, request
from app.services.settings_service import SettingsService, Thresholds

bp = Blueprint("settings", __name__)
_svc = SettingsService()

@bp.route("/thresholds", methods=["GET"])
def get_thresholds():
    t = _svc.load()
    return jsonify({"conf_th": t.conf_th, "iou_th": t.iou_th, "min_box_frac": t.min_box_frac})

@bp.route("/thresholds", methods=["POST"])
def set_thresholds():
    data = request.get_json(force=True, silent=True) or {}
    t = _svc.load()
    if "conf_th" in data:      t.conf_th = float(data["conf_th"])
    if "iou_th" in data:       t.iou_th = float(data["iou_th"])
    if "min_box_frac" in data: t.min_box_frac = float(data["min_box_frac"])
    _svc.save(t)
    return jsonify({"ok": True})

@bp.route("/thresholds/reset", methods=["POST"])
def reset_thresholds():
    t = _svc.reset()
    return jsonify({"conf_th": t.conf_th, "iou_th": t.iou_th, "min_box_frac": t.min_box_frac})