import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor

app = Flask(__name__)

# ==========================================
# 1. TRAIN MULTI-DYE MODEL (5 DYES SYSTEM)
# ==========================================
np.random.seed(42)
n_samples = 500

# CIELAB & Kubelka-Munk Synthetic Training Set
L_star = np.random.uniform(10, 90, n_samples)
a_star = np.random.uniform(-30, 70, n_samples)
b_star = np.random.uniform(-30, 70, n_samples)

R_val = np.clip(
    (L_star / 100.0) ** 2.2 + np.random.normal(0, 0.005, n_samples), 0.005, 0.98
)
KS_val = ((1.0 - R_val) ** 2) / (2.0 * R_val)

# Formulation Rules for 5 Dyes
Dye_Black = np.clip(0.12 * KS_val * (L_star < 35), 0, 4.0)
Dye_Red = np.clip(0.04 * KS_val * (a_star > 10) * (a_star / 40.0), 0, 3.0)
Dye_Blue = np.clip(
    0.04 * KS_val * (b_star < -5) * (abs(b_star) / 35.0), 0, 3.0
)
Dye_Yellow = np.clip(0.05 * KS_val * (b_star > 15) * (b_star / 40.0), 0, 3.0)
Dye_Orange = np.clip(
    0.03 * KS_val * (a_star > 5) * (b_star > 5) * ((a_star + b_star) / 60.0),
    0,
    2.5,
)

X = np.column_stack([L_star, a_star, b_star, R_val, KS_val])
Y = np.column_stack([Dye_Red, Dye_Blue, Dye_Yellow, Dye_Black, Dye_Orange])

# Multi-Output Model Training
base_model = ExtraTreesRegressor(
    n_estimators=150, max_depth=6, random_state=42
)
model = MultiOutputRegressor(base_model)
model.fit(X, Y)


# ==========================================
# 2. FEATURE EXTRACTION (OpenCV & CIELAB)
# ==========================================
def extract_hybrid_features(img_bytes):
  nparr = np.frombuffer(img_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  if img is None:
    return None

  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  L_star_val = float(np.mean(lab[:, :, 0])) * (100.0 / 255.0)
  a_star_val = float(np.mean(lab[:, :, 1])) - 128.0
  b_star_val = float(np.mean(lab[:, :, 2])) - 128.0

  R = np.clip((L_star_val / 100.0) ** 2.2, 0.001, 0.999)
  KS_value = ((1.0 - R) ** 2) / (2.0 * R)

  return np.array([L_star_val, a_star_val, b_star_val, R, KS_value])


# ==========================================
# 3. ROUTES
# ==========================================
@app.route('/')
def index():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  if 'fabric_image' not in request.files:
    return jsonify({'error': 'Hakuna picha iliyopakiwa!'}), 400

  file = request.files['fabric_image']
  fabric_weight = float(request.form.get('fabric_weight', 8.0))
  liquor_ratio = float(request.form.get('liquor_ratio', 10.0))

  if file.filename == '':
    return jsonify({'error': 'Tafadhali chagua picha'}), 400

  img_bytes = file.read()
  features = extract_hybrid_features(img_bytes)

  if features is None:
    return jsonify({'error': 'Imeshindwa kusoma picha'}), 400

  predicted = model.predict([features])[0]
  scale = fabric_weight / 100.0

  # Kiasi halisi kisicho na mduara (float)
  raw_red = max(0.0, float(predicted[0]) * scale)
  raw_blue = max(0.0, float(predicted[1]) * scale)
  raw_yellow = max(0.0, float(predicted[2]) * scale)
  raw_black = max(0.0, float(predicted[3]) * scale)
  raw_orange = max(0.0, float(predicted[4]) * scale)

  # 1. Kiasi kwa Gramu kilichowekewa Decimal Places 2 (mzani wa 0.01g)
  red_g = round(raw_red, 2)
  blue_g = round(raw_blue, 2)
  yellow_g = round(raw_yellow, 2)
  black_g = round(raw_black, 2)
  orange_g = round(raw_orange, 2)

  # 2. Kiasi cha Stock Solution (1% liquid = 0.01g/mL) kwa ajili ya Syringe/Pipette
  red_ml = round(raw_red / 0.01, 1)
  blue_ml = round(raw_blue / 0.01, 1)
  yellow_ml = round(raw_yellow / 0.01, 1)
  black_ml = round(raw_black / 0.01, 1)
  orange_ml = round(raw_orange / 0.01, 1)

  water_ml = fabric_weight * liquor_ratio

  return jsonify({
      'success': True,
      'ks_value': round(float(features[4]), 4),
      'l_star': round(float(features[0]), 2),
      'a_star': round(float(features[1]), 2),
      'b_star': round(float(features[2]), 2),
      'red_g': f'{red_g:.2f}',
      'blue_g': f'{blue_g:.2f}',
      'yellow_g': f'{yellow_g:.2f}',
      'black_g': f'{black_g:.2f}',
      'orange_g': f'{orange_g:.2f}',
      'red_ml': red_ml,
      'blue_ml': blue_ml,
      'yellow_ml': yellow_ml,
      'black_ml': black_ml,
      'orange_ml': orange_ml,
      'water_ml': round(water_ml, 1),
  })


if __name__ == '__main__':
  app.run(debug=True, port=5000)