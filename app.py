from flask import Flask, request, jsonify
from mediapipe.tasks import python
from mediapipe.tasks.python import text

app = Flask(__name__)

base_options = python.BaseOptions(model_asset_path="language_detector.tflite")
options = text.LanguageDetectorOptions(base_options=base_options)
detector = text.LanguageDetector.create_from_options(options)
@app.route('/')
def index():
    return "Hello Shivain"
@app.route('/detect_language', methods=['POST'])
def detect_language():
    input_text = request.json.get('text')
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    detection_result = detector.detect(input_text)

    detections = [
        {"language_code": detection.language_code, "probability": detection.probability}
        for detection in detection_result.detections
    ]

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True)