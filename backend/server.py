from flask import Flask, request, jsonify
from tensorflow.python.keras.models import load_model

app = Flask(__name__)