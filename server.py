from flask import Flask, send_file, request, flash, jsonify
import cosfire as c
import time
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('html/index.html')

@app.route('/fit', methods=['POST'])
def fit():
    ''' Arguments '''
    proto = np.asarray(Image.open(request.files['prototype']).convert('L'), dtype=np.float64)
    print(proto.shape)
    cx = request.form.get('protoX', int(proto.shape[1]/2), type=int)
    cy = request.form.get('protoY', int(proto.shape[0]/2), type=int)
    minRho = request.form.get('minRho', 0, type=int)
    maxRho = request.form.get('maxRho', 10, type=int)
    stepRho = request.form.get('stepRho', 2, type=int)
    sigma = request.form.get('sigma', 2, type=float)
    onoff = request.form.get('onoff', 1, type=int)
    sigma0 = request.form.get('sigma0', 1, type=float)
    alpha = request.form.get('alpha', 0.1, type=float)
    rotation = request.form.get('rotation', 'full', type=str)
    rotationInvariance = np.arange(24)/24 if rotation=='full' else (np.arange(12)/24 if rotation=='half' else [0])

    ''' Run COSFIRE '''
    cosfire = c.COSFIRE(
    		c.CircleStrategy, c.DoGFilter, (sigma, onoff), rhoList=range(minRho,maxRho,stepRho), sigma0=sigma0,  alpha=alpha,
    		rotationInvariance = rotationInvariance*np.pi, numthreads = 4
    	   ).fit(proto, (cx, cy))
    return jsonify({"tuples": cosfire.strategy.tuples})

if __name__ == "__main__":
    app.run(debug=True)
