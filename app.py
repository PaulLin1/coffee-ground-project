from flask import Flask, flash, redirect, render_template, session, url_for, request, jsonify
from main import *
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        filename = "original.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        file.save(file_path)
        
        no_clumps_path = remove_clumps(file_path)

        black_and_white(file_path)
        black_and_white(no_clumps_path)

        return redirect(url_for('index'))

@app.route('/calculate-areas', methods=['GET', 'POST'])
def calculate_areas_path():
    pixels_to_unit = request.args.get('pixels_to_unit')

    return calculate_areas(pixels_to_unit)

if __name__ == '__main__':
    app.run(debug=True)