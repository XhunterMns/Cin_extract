import os
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from werkzeug.utils import secure_filename
app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadForm(FlaskForm):
    file = FileField('File')
    submit = SubmitField('upload file')

@app.route('/')
@app.route('/home', methods=['GET','POST'])


def home():
    form = UploadForm()
    # Ensure upload folder exists (absolute path)
    upload_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])
    os.makedirs(upload_dir, exist_ok=True)

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        if filename:
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)

    # List files (only regular files) to pass to template
    try:
        files = sorted([f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))])
    except FileNotFoundError:
        files = []

    return render_template('index.html', form=form, files=files)
  
def show_static_files():
    # Path relative to your project root
    dir_path = os.path.join(app.static_folder, 'files')

    if not os.path.isdir(dir_path):
        files = []
    else:
        files = sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

    # Render the same index template (or a dedicated files.html if you create one)
    return render_template('index.html', files=files, form=UploadForm())



if __name__ == '__main__':
    app.run(debug=True)