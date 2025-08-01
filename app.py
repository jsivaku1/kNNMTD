import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

def prepare_data(df, fit_encoder=None):
    """Pipeline Step 1: Prepares raw data for the algorithm."""
    df_processed = df.copy()
    
    # Impute missing values
    for col in df_processed.columns:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        else:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    # Convert object columns to category for one-hot encoding
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = df_processed[col].astype('category')
    
    # One-hot encode categorical features
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    return df_processed, {} # Return empty inversion map for now

def postprocess_data(synthetic_df, original_df, inversion_maps):
    """Pipeline Step 3: Converts generated data back to original format."""
    df_final = synthetic_df.copy()
    for col, mapping in inversion_maps.items():
        inverse_map = {v: k for k, v in mapping.items()}
        df_final[col] = df_final[col].round().astype(int).map(inverse_map)
    for col in original_df.columns:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(original_df[col].dtype)
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
            return jsonify({"error": "Could not process CSV file."}), 500
    return jsonify({"error": "Invalid file type"}), 400

def run_ml_utility_test(real_df_raw, synthetic_df_final, target_col_name, task_mode):
    if target_col_name not in real_df_raw.columns or synthetic_df_final.empty:
        return {}
    X_real, y_real = real_df_raw.drop(columns=[target_col_name]), real_df_raw[target_col_name]
    X_synth, y_synth = synthetic_df_final.drop(columns=[target_col_name]), synthetic_df_final[target_col_name]
    
    X_train, _ = prepare_data(X_synth)
    y_train = y_synth
    X_test, _ = prepare_data(X_real)
    y_test = y_real
    
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test: X_test[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train: X_train[c] = 0
    X_test = X_test[train_cols]
    
    metrics = {}
    
    if task_mode == 'classification':
        model = RandomForestClassifier(random_state=42, n_estimators=50).fit(X_train, y_train)
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
        model = RandomForestRegressor(random_state=42, n_estimators=50).fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics['r2_score'] = r2_score(y_test, preds)
        metrics['mae'] = mean_absolute_error(y_test, preds)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, preds))
        
    return {k: round(v, 4) for k, v in metrics.items()}

def run_unsupervised_utility_test(real_df_processed, synthetic_df_processed):
    if real_df_processed.empty or synthetic_df_processed.empty:
        return {}
    
    real_cols = real_df_processed.columns
    synth_cols = synthetic_df_processed.columns
    missing_in_synth = set(real_cols) - set(synth_cols)
    for c in missing_in_synth: synthetic_df_processed[c] = 0
    synthetic_df_processed = synthetic_df_processed[real_cols]

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df_processed)
    synth_scaled = scaler.transform(synthetic_df_processed)
    
    n_clusters = 5 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(real_scaled)
    
    real_clusters = pd.Series(kmeans.predict(real_scaled))
    synth_clusters = pd.Series(kmeans.predict(synth_scaled))
    
    real_dist = real_clusters.value_counts(normalize=True).sort_index()
    synth_dist = synth_clusters.value_counts(normalize=True).sort_index()
    
    dist_df = pd.DataFrame({'Real': real_dist, 'Synthetic': synth_dist}).fillna(0) * 100
    
    return {f"Cluster {i} (%)": [round(dist_df.loc[i, 'Real'],2), round(dist_df.loc[i, 'Synthetic'],2)] for i in dist_df.index}

def get_pca_data(real_df_processed, synthetic_df_processed):
    if real_df_processed.empty or synthetic_df_processed.empty:
        return {}
        
    real_cols = real_df_processed.columns
    synth_cols = synthetic_df_processed.columns
    missing_in_synth = set(real_cols) - set(synth_cols)
    for c in missing_in_synth: synthetic_df_processed[c] = 0
    synthetic_df_processed = synthetic_df_processed[real_cols]

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df_processed)
    synth_scaled = scaler.transform(synthetic_df_processed)

    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_scaled)
    synth_pca = pca.transform(synth_scaled)

    return {
        'real': pd.DataFrame(real_pca, columns=['x', 'y']).to_dict('records'),
        'synthetic': pd.DataFrame(synth_pca, columns=['x', 'y']).to_dict('records')
    }


@app.route('/generate', methods=['POST'])
def generate_synthetic_data():
    try:
        start_time = time.time()
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
            
            real_df_processed, inversion_maps = prepare_data(real_df_raw)
            task_mode = 'unsupervised'
            if class_col_name:
                y_raw = real_df_raw[class_col_name]
                analysis = analyze_columns(y_raw.to_frame())
                task_mode = 'classification' if analysis['categorical'] else 'regression'
            
            max_k = len(real_df_raw) - 1
            if task_mode == 'classification':
                min_class_size = real_df_raw[class_col_name].value_counts().min()
                max_k = min_class_size
            
            k_options = [k for k in [2, 3, 5] if k <= max_k]
            if not k_options: k_options = [2]
            
            best_k, best_pcd = -1, float('inf')
            
            for k_val in k_options:
                n_obs_test = min(20, int(request.form['n_obs']))
                X, y = real_df_processed.copy(), None
                if class_col_name: y = X.pop(class_col_name)
                model = kNNMTD(n_obs=n_obs_test, k=k_val, n_epochs=1)
                final_df_for_k = next(model.fit_generate(X, y), None)
                if final_df_for_k is not None and not final_df_for_k.empty:
                    current_pcd = PCD(real_df_processed, final_df_for_k)
                    if not np.isnan(current_pcd) and current_pcd < best_pcd:
                        best_pcd = current_pcd
                        best_k = k_val
            
            if best_k == -1:
                return render_template('error.html', error_message="Failed to generate data for any k value.")

            n_obs = int(request.form['n_obs'])
            X, y = real_df_processed.copy(), None
            if class_col_name: y = X.pop(class_col_name)
            model = kNNMTD(n_obs=n_obs, k=best_k, n_epochs=n_epochs)
            pcd_over_time, ml_metrics_over_time, final_synthetic_df = [], {}, None
            
            for synthetic_batch_processed in model.fit_generate(X, y):
                synthetic_batch_final = postprocess_data(synthetic_batch_processed, real_df_raw, inversion_maps)
                pcd_over_time.append(PCD(real_df_raw, synthetic_batch_final))
                
                if task_mode != 'unsupervised':
                    metrics = run_ml_utility_test(real_df_raw, synthetic_batch_final, class_col_name, task_mode)
                else:
                    metrics = run_unsupervised_utility_test(real_df_processed, synthetic_batch_processed)

                for key, value in metrics.items():
                    ml_metrics_over_time.setdefault(key, []).append(value)
                final_synthetic_df = synthetic_batch_final

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_filename = secure_filename(file.filename).rsplit('.', 1)[0]
            synthetic_filename = f"{base_filename}_sd_{timestamp}.csv"
            final_synthetic_df.to_csv(os.path.join(app.config['GENERATED_FOLDER'], synthetic_filename), index=False)
            
            total_runtime = round(time.time() - start_time, 2)
            
            final_synthetic_df_processed, _ = prepare_data(final_synthetic_df)
            pca_data = get_pca_data(real_df_processed, final_synthetic_df_processed)

            initial_metrics = {"PCD": pcd_over_time[0] if pcd_over_time else "N/A"}
            final_metrics = {"PCD": pcd_over_time[-1] if pcd_over_time else "N/A"}
            for key, values in ml_metrics_over_time.items():
                initial_metrics[key] = values[0]
                final_metrics[key] = values[-1]

            return render_template('results.html', 
                                   pcd_scores=pcd_over_time, 
                                   ml_metrics_over_time=ml_metrics_over_time,
                                   generated_filename=synthetic_filename,
                                   initial_metrics=initial_metrics,
                                   final_metrics=final_metrics,
                                   pca_data=pca_data,
                                   pipeline_stages=[
                                       ("Data Preparation", "Cleaned & encoded data"),
                                       ("K-Optimization", f"Best k found: {best_k}"),
                                       ("Generation", f"Created {n_obs} samples over {n_epochs} epochs"),
                                       ("Post-processing", "Converted data to original format")
                                   ],
                                   input_options={
                                       "Dataset": secure_filename(file.filename),
                                       "Real Data": f"{len(real_df_raw)} rows, {len(real_df_raw.columns)} columns",
                                       "Synthetic Data": f"{len(final_synthetic_df)} rows, {len(final_synthetic_df.columns)} columns",
                                       "Task": task_mode.capitalize(),
                                       "Target": class_col_name if class_col_name else "None",
                                       "Total Runtime": f"{total_runtime} seconds"
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