import os
import logging
import datetime
import jwt
from functools import wraps

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker


JWT_SECRET= "MYSECRETKEY"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600  # 1 hour


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")  

DB_URL =  "sqlite:///predictions.db"
engine = create_engine(DB_URL, echo=True)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

model = joblib.load("modelo_iris.pkl")
logger.info("Modelo cargado correctamente.")


app = Flask(__name__)
predictions_cache = {}
TEST_USERNAME = "admin"  
TEST_PASSWORD = "secret"


def create_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # pegar token do header Authorization: Bearer <token>
        # decodificar e checar expiração
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")

    if username == TEST_USERNAME and password == TEST_PASSWORD:
        token = create_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"error": "Credenciais inválidas"}), 401

@app.route("/predict", methods=["POST"])
@token_required
def predict():
    data = request.get_json(force=True)
    try:
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    except (ValueError, KeyError) as e:
        logger.error("Dados de entrada inválidos: %s", e)
        return jsonify({"error": "Dados inválidos, verifique parâmetros"}), 400

    features = (sepal_length, sepal_width, petal_length, petal_width)
    if features in predictions_cache:
        logger.info("Cache hit para %s", features)
        predicted_class = predictions_cache[features]
    else:
        input_data = np.array([features])
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])
        predictions_cache[features] = predicted_class
        logger.info("Cache updated para %s", features)

    db = SessionLocal()
    new_pred = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class
    )
    db.add(new_pred)
    db.commit()
    db.close()

    return jsonify({
        "predicted_class": predicted_class,
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }), 200


@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    """
    Lista as predições armazenadas no banco.
    Parâmetros opcionais (via query string):
    - limit (int): quantos registros retornar, padrão 10
    - offset (int): a partir de qual registro começar, padrão 0

    Exemplo:
    /predictions?limit=5&offset=10
    """
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))
    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()
    db.close()
    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "sepal_length": p.sepal_length,
            "sepal_width": p.sepal_width,
            "petal_length": p.petal_length,
            "petal_width": p.petal_width,
            "predicted_class": p.predicted_class,
            "created_at": p.created_at.isoformat()
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)