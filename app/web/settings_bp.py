# app/web/settings.py
from flask import Blueprint, jsonify, request
from app.services.settings_service import THRESHOLDS_CACHE

bp = Blueprint("settings", __name__)

@bp.route("/thresholds", methods=["GET"])
def get_thresholds():
    t, _ver = THRESHOLDS_CACHE.snapshot()
    return jsonify({"conf_th": t.conf_th, "iou_th": t.iou_th, "min_box_frac": t.min_box_frac})

@bp.route("/thresholds", methods=["POST"])
def set_thresholds():
    data = request.get_json(force=True, silent=True) or {}
    t = THRESHOLDS_CACHE.update(**data)
    return jsonify({"conf_th": t.conf_th, "iou_th": t.iou_th, "min_box_frac": t.min_box_frac})

@bp.route("/thresholds/reset", methods=["POST"])
def reset_thresholds():
    t = THRESHOLDS_CACHE.reset()
    return jsonify({"conf_th": t.conf_th, "iou_th": t.iou_th, "min_box_frac": t.min_box_frac})