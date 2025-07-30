import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import time
import logging
import traceback

# Import your custom modules
from kNNMTD import kNNMTD
from utils import PCD

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Create directories if they don't exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# --- Set up Logging ---
# This will create an app.log file in your project directory
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the main page with the upload form."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_synthetic_data():
    """Handles the form submission, runs the kNNMTD model, and shows results."""
    try:
        # --- 1. Handle File Upload ---
        if 'dataset' not in request.files:
            app.logger.warning("No file part in the request.")
            return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '':
            app.logger.warning("No selected file.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            real_data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(real_data_path)
            app.logger.info(f"File '{filename}' uploaded successfully.")

            # --- 2. Get Form Parameters ---
            n_obs = int(request.form['n_obs'])
            k = int(request.form['k_value'])
            mode = int(request.form['mode'])
            class_col = request.form.get('class_col', None)

            # If class_col is an empty string, treat it as None
            if not class_col:
                class_col = None

            if mode in [0, 1] and not class_col:
                app.logger.error("Form validation error: Class column required for Classification/Regression but not provided.")
                return render_template('error.html', error_message="Class/Target column is required for Classification or Regression.")

            app.logger.info(f"Parameters received: n_obs={n_obs}, k={k}, mode={mode}, class_col='{class_col}'")

            # --- 3. Run the kNNMTD Model ---
            real_df = pd.read_csv(real_data_path)

            if class_col and class_col not in real_df.columns:
                 app.logger.error(f"Column '{class_col}' not found in the uploaded dataset.")
                 return render_template('error.html', error_message=f"Error: Column '{class_col}' not found in the uploaded dataset.")

            model = kNNMTD(n_obs=n_obs, k=k, mode=mode)

            app.logger.info("Starting kNNMTD model fitting...")
            if mode != -1:
                synthetic_df = model.fit(real_df, class_col=class_col)
            else:
                synthetic_df = model.fit(real_df) # Unsupervised case
            app.logger.info("kNNMTD model fitting complete.")

            app.logger.info("Calculating PCD score...")
            pcd_score = PCD(real_df, synthetic_df)
            app.logger.info(f"PCD score calculated: {pcd_score}")


            # --- 4. Save Synthetic Data ---
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            synthetic_filename = f"synthetic_{timestamp}.csv"
            synthetic_data_path = os.path.join(app.config['GENERATED_FOLDER'], synthetic_filename)
            synthetic_df.to_csv(synthetic_data_path, index=False)
            app.logger.info(f"Synthetic data saved to '{synthetic_filename}'.")

            return render_template('results.html', pcd_score=pcd_score, generated_filename=synthetic_filename)

        else:
            app.logger.warning(f"File with disallowed extension uploaded: {file.filename}")
            return render_template('error.html', error_message="Invalid file type. Please upload a .csv file.")

    except Exception as e:
        # Log the full error traceback to the app.log file
        error_traceback = traceback.format_exc()
        app.logger.error(f"An unexpected error occurred: {e}\n{error_traceback}")
        # Return a user-friendly error page
        return render_template('error.html', error_message="An internal error occurred. The details have been logged for the administrator.")


@app.route('/download/<filename>')
def download_file(filename):
    """Serves the generated file for download."""
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
