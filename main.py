import os
import time
import base64
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, Response
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from camera_local import VideoCamera
import ocr  # use functions from your ocr.py

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadForm(FlaskForm):
    file = FileField('File')
    submit = SubmitField('upload file')

def get_upload_path():
    """Helper to get absolute path to upload folder"""
    base_dir = os.path.abspath(os.path.dirname(__file__))
    upload_dir = os.path.join(base_dir, app.config['UPLOAD_FOLDER'])
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    upload_dir = get_upload_path()

    # LOGIC FOR FILE UPLOAD
    if form.validate_on_submit():
        file = form.file.data
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(upload_dir, filename))
            flash(f'File {filename} uploaded successfully!')
            return redirect(url_for('home'))

    # LOGIC FOR LISTING FILES
    files = sorted([f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))])
    
    return render_template('index.html', form=form, files=files)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route('/capture', methods=['POST'])
def capture():
    upload_dir = get_upload_path()

    try:
        cam = VideoCamera()
        frame_bytes = cam.get_frame() 
        
        if not frame_bytes:
            flash('Failed to capture image from camera.')
            return redirect(url_for('home'))

        filename = f"cin_{int(time.time())}.jpg"
        save_path = os.path.join(upload_dir, filename)
        
        with open(save_path, 'wb') as f:
            f.write(frame_bytes)

        flash(f"Saved photo as {filename}")
    except Exception as e:
        flash(f"Error capturing photo: {e}")

    return redirect(url_for('home'))

@app.route('/extract', methods=['POST'])
def extract():
    """
    Accept multipart 'file' or JSON { data_url } or { filename }.
    If client sets debug=1 (form field or JSON flag), the server returns
    'preprocessed_image' (data URL) as part of the JSON response.
    """
    debug = False
    if 'debug' in request.form:
        debug = request.form.get('debug') in ('1', 'true', 'True')
    if request.is_json:
        js = request.get_json(silent=True) or {}
        if js.get("debug") in (True, "1", "true", "True"):
            debug = True

    # 1) multipart file
    if 'file' in request.files:
        f = request.files['file']
        data = f.read()
        try:
            result = ocr.extract_text_from_image(data, debug=debug)
            text = result.get("text", "")
            fields = ocr.extract_fields(text)
            meta = {k: result.get(k) for k in ("variant","lang","psm","confidence","region","region_score")}
            resp = {"text": text, "fields": fields, "meta": meta}
            if debug and result.get("preprocessed_image"):
                resp["preprocessed_image"] = result["preprocessed_image"]
            return jsonify(resp)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # 2) JSON body (data_url or filename)
    if request.is_json:
        payload = request.get_json()
        # data_url
        if payload.get("data_url"):
            header, b64 = payload["data_url"].split(",", 1) if "," in payload["data_url"] else (None, payload["data_url"])
            try:
                img_bytes = base64.b64decode(b64)
            except Exception:
                return jsonify({"error": "invalid base64 data"}), 400
            try:
                result = ocr.extract_text_from_image(img_bytes, debug=debug)
                text = result.get("text", "")
                fields = ocr.extract_fields(text)
                meta = {k: result.get(k) for k in ("variant","lang","psm","confidence","region","region_score")}
                resp = {"text": text, "fields": fields, "meta": meta}
                if debug and result.get("preprocessed_image"):
                    resp["preprocessed_image"] = result["preprocessed_image"]
                return jsonify(resp)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        # filename (legacy)
        if payload.get("filename"):
            safe = secure_filename(payload.get("filename"))
            upload_dir = get_upload_path()
            filepath = os.path.join(upload_dir, safe)
            if not os.path.isfile(filepath):
                return jsonify({"error": "file not found"}), 404
            try:
                result = ocr.extract_text_from_image(filepath, debug=debug)
                text = result.get("text", "")
                fields = ocr.extract_fields(text)
                meta = {k: result.get(k) for k in ("variant","lang","psm","confidence","region","region_score")}
                resp = {"text": text, "fields": fields, "meta": meta}
                if debug and result.get("preprocessed_image"):
                    resp["preprocessed_image"] = result["preprocessed_image"]
                return jsonify(resp)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    return jsonify({"error": "no image provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)