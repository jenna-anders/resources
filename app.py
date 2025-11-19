from flask import Flask

app = Flask(__name__)

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/")
def index():
    return "OK", 200

@app.route("/ok")
def ok():
    return "OK", 200
