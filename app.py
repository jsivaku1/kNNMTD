import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
import numpy as np

# Import your custom modules
from kNNMTD import kNNMTD
from utils import PCD, analyze_columns

# --- Flask App Initialization & Config ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
LOG_FILE = 'app.log'
app.config.from_object(__name__)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
handler = RotatingFileHandler(LOG_FILE, maxBytes=2*1024*1024, backupCount=3)
handler.setFormatter(log_formatter)
app.logger.handlers.clear()
logging.getLogger('werkzeug').handlers.clear()
app.logger.addHandler(handler)
logging.getLogger('werkzeug').addHandler(handler)
app.logger.setLevel(logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def prepare_data(df):
    """
    Pipeline Step 1: Prepares raw data for the algorithm.
    - Imputes missing values.
    - Converts categorical columns to numerical codes.
    Returns the processed DataFrame and a dictionary for inverse transformation.
    """
    app.logger.info("Pipeline Stage 1: Preparing Data...")
    df_processed = df.copy()
    inversion_maps = {}

    for col in df_processed.columns:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                app.logger.info(f"  - Imputed missing in '{col}' with median ({median_val}).")
        else: # Categorical or object
            if df_processed[col].isnull().any():
                mode_val = df_processed[col].mode()[0]
                df_processed[col].fillna(mode_val, inplace=True)
                app.logger.info(f"  - Imputed missing in '{col}' with mode ('{mode_val}').")
            
            df_processed[col] = df_processed[col].astype('category')
            inversion_maps[col] = dict(enumerate(df_processed[col].cat.categories))
            df_processed[col] = df_processed[col].cat.codes
            app.logger.info(f"  - Converted categorical column '{col}' to numerical codes.")
            
    app.logger.info("Data preparation complete.")
    return df_processed, inversion_maps

def postprocess_data(synthetic_df, original_df, inversion_maps):
    """
    Pipeline Step 3: Converts the generated numerical data back to its original format.
    """
    app.logger.info("Pipeline Stage 3: Post-processing Data...")
    df_final = synthetic_df.copy()

    for col, mapping in inversion_maps.items():
        inverse_map = {v: k for k, v in mapping.items()}
        # Round to nearest integer code before mapping back
        df_final[col] = df_final[col].round().astype(int).map(inverse_map)
        app.logger.info(f"  - Converted column '{col}' back to original categories.")

    for col in original_df.columns:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(original_df[col].dtype)
    
    app.logger.info("Post-processing complete.")
    return df_final


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/info', methods=['POST'])
def get_file_info():
    if 'dataset' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['dataset']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)
            column_info = analyze_columns(df)
            return jsonify(column_info)
        except Exception as e:
            app.logger.error(f"Error reading CSV for info: {e}")
            return jsonify({"error": "Could not process CSV file."}), 500
    return jsonify({"error": "Invalid file type"}), 400

def run_ml_utility_test(real_df_raw, synthetic_df_final, target_col_name, task_mode):
    if target_col_name not in real_df_raw.columns or target_col_name not in synthetic_df_final.columns or synthetic_df_final.empty:
        return {}
        
    # Prepare data for ML model (needs to be numeric)
    X_real, _ = prepare_data(real_df_raw.drop(columns=[target_col_name]))
    y_real = real_df_raw[target_col_name]
    
    X_test_synth, _ = prepare_data(synthetic_df_final.drop(columns=[target_col_name]))
    y_test_synth = synthetic_df_final[target_col_name]
    
    X_train, y_train = X_real, y_real
    X_test, y_test = X_test_synth, y_test_synth

    metrics = {}
    
    if task_mode == 'classification':
        model = RandomForestClassifier(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics['accuracy'] = accuracy_score(y_test, preds)
        metrics['f1_score'] = f1_score(y_test, preds, average='weighted')
        if len(y_train.unique()) > 1 and len(y_test.unique()) > 1:
            try:
                preds_proba = model.predict_proba(X_test)
                if len(y_train.unique()) == 2 and preds_proba.shape[1] == 2:
                     metrics['auc'] = roc_auc_score(y_test, preds_proba[:, 1])
                elif len(y_train.unique()) > 2:
                     metrics['auc'] = roc_auc_score(y_test, preds_proba, multi_class='ovr', average='weighted')
            except Exception: pass
    elif task_mode == 'regression':
        model = RandomForestRegressor(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics['r2_score'] = r2_score(y_test, preds)
    return {k: round(v, 4) for k, v in metrics.items()}

@app.route('/generate', methods=['POST'])
def generate_synthetic_data():
    try:
        if 'dataset' not in request.files: return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '': return redirect(request.url)

        if file and allowed_file(file.filename):
            real_data_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.seek(0); file.save(real_data_path)
            
            class_col_name = request.form.get('class_col', None)
            n_epochs = int(request.form.get('n_epochs', 10))
            if not class_col_name or class_col_name == "none": class_col_name = None

            real_df_raw = pd.read_csv(real_data_path)
            
            # --- Pipeline Stage 1: Data Prep ---
            real_df_processed, inversion_maps = prepare_data(real_df_raw)
            task_mode = 'unsupervised'
            if class_col_name:
                y_raw = real_df_raw[class_col_name]
                analysis = analyze_columns(y_raw.to_frame())
                if analysis['categorical']: task_mode = 'classification'
                else: task_mode = 'regression'
            
            app.logger.info(f"Detected task mode: {task_mode}")
            
            # --- Pipeline Stage 2: K-Optimization ---
            max_k = len(real_df_raw) - 1
            if task_mode == 'classification':
                min_class_size = real_df_raw[class_col_name].value_counts().min()
                max_k = min_class_size
            
            k_options = [k for k in [2, 3, 5] if k <= max_k]
            if not k_options: k_options = [2]
            
            best_k, best_pcd = -1, float('inf')
            
            app.logger.info(f"Starting k optimization. Trying k values: {k_options}")
            for k_val in k_options:
                n_obs_test = min(20, int(request.form['n_obs']))
                X, y = real_df_processed.copy(), None
                if class_col_name: y = X.pop(class_col_name)

                model = kNNMTD(n_obs=n_obs_test, k=k_val, n_epochs=1)
                final_df_for_k = None
                for synthetic_batch in model.fit_generate(X, y):
                    final_df_for_k = synthetic_batch
                
                if final_df_for_k is not None and not final_df_for_k.empty:
                    current_pcd = PCD(real_df_processed, final_df_for_k)
                    if np.isnan(current_pcd): continue
                    app.logger.info(f"Tested k={k_val}, final PCD={current_pcd}")
                    if current_pcd < best_pcd:
                        best_pcd = current_pcd
                        best_k = k_val
            
            if best_k == -1:
                return render_template('error.html', error_message="Failed to generate data for any k value. The dataset may be too small.")

            notification = f"Automatic Optimization: Best 'k' value found to be {best_k} (out of {k_options}) with a test PCD of {round(best_pcd, 4)}."
            
            # --- Pipeline Stage 3: Final Generation ---
            n_obs = int(request.form['n_obs'])
            X, y = real_df_processed.copy(), None
            if class_col_name: y = X.pop(class_col_name)
            model = kNNMTD(n_obs=n_obs, k=best_k, n_epochs=n_epochs)
            pcd_over_time, ml_metrics_over_time, final_synthetic_df = [], {}, None
            
            app.logger.info(f"Pipeline Stage 2: Generating data with optimal k={best_k}...")
            for synthetic_batch_processed in model.fit_generate(X, y):
                synthetic_batch_final = postprocess_data(synthetic_batch_processed, real_df_raw, inversion_maps)
                
                pcd_over_time.append(PCD(real_df_raw, synthetic_batch_final))
                if task_mode != 'unsupervised':
                    metrics = run_ml_utility_test(real_df_raw, synthetic_batch_final, class_col_name, task_mode)
                    for key, value in metrics.items():
                        ml_metrics_over_time.setdefault(key, []).append(value)
                final_synthetic_df = synthetic_batch_final

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_filename = secure_filename(file.filename).rsplit('.', 1)[0]
            synthetic_filename = f"{base_filename}_sd_{timestamp}.csv"
            final_synthetic_df.to_csv(os.path.join(app.config['GENERATED_FOLDER'], synthetic_filename), index=False)
            app.logger.info(f"Process complete. Final data saved to '{synthetic_filename}'.")

            # Prepare final metrics for display
            final_metrics = {
                "Final PCD": pcd_over_time[-1] if pcd_over_time else "N/A"
            }
            for key, values in ml_metrics_over_time.items():
                final_metrics[f"Final {key}"] = values[-1]

            return render_template('results.html', 
                                   pcd_scores=pcd_over_time, 
                                   ml_metrics_over_time=ml_metrics_over_time,
                                   generated_filename=synthetic_filename,
                                   final_metrics=final_metrics,
                                   pipeline_stages=[
                                       f"1. Data Preparation: Cleaned and encoded data.",
                                       notification,
                                       f"3. Generation: Created {n_obs} samples over {n_epochs} epochs.",
                                       f"4. Post-processing: Converted synthetic data back to original format."
                                   ],
                                   input_options={
                                       "Task": task_mode.capitalize(),
                                       "Target Column": class_col_name if class_col_name else "None",
                                       "Optimal k": best_k,
                                       "Epochs": n_epochs,
                                       "Samples Generated": n_obs
                                   })
        else:
            return render_template('error.html', error_message="Invalid file type.")

    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {traceback.format_exc()}")
        return render_template('error.html', error_message="An internal error occurred. Details have been logged.")

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
