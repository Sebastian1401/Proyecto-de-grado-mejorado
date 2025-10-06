import argparse
from rknn.api import RKNN

ap = argparse.ArgumentParser()
ap.add_argument('--onnx', required=True)
ap.add_argument('--out',  required=True)
args = ap.parse_args()

r = RKNN(verbose=True)
r.config(
    target_platform='rk3588',
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],   # normaliza a [0,1]
    quant_img_RGB2BGR=False,        # mantenemos RGB
    output_optimize=True            # <-- bool, no string
)

assert r.load_onnx(args.onnx) == 0
assert r.build(do_quantization=False) == 0
assert r.export_rknn(args.out) == 0
r.release()
print('OK')
