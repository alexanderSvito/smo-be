from flask import Flask, jsonify
from smo import QueueingSystem
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/run')
def hello_world():
    smo = QueueingSystem(
        "M/M/1",
        infinite_queue=True,
        # T='M',
        # T_params=(3,),
        A_params=(15,),
        B_params=(5,)
    )
    gen = smo.start()
    result = []
    for data in gen:
        result.append(data)
    return jsonify(result)
