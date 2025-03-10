# import mlflow.pyfunc
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# # Load trained model from MLflow
# model = mlflow.pyfunc.load_model("mlruns/0/xgboost_model")

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Accepts JSON input and returns predictions."""
#     data = request.get_json()
#     predictions = model.predict([data["features"]])
#     return jsonify({"prediction": predictions.tolist()})

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5000)