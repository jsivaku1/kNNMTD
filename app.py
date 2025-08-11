import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, roc_auc_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from dython.nominal import associations

# Import your custom modules
from kNNMTD import kNNMTD
from utils import PCD, analyze_columns

# --- Flask App Initialization & Config ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
LOG_FILE = 'app.log'
app.config.from_object(__name__)
app.config['BIG_DATA_CELL_THRESHOLD'] = 500000 
app.config['DEFAULT_SAMPLE_SIZE'] = 5000

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
    df_processed = df.copy()
    for col in df_processed.columns:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        else:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    for col in df_processed.select_dtypes(include=['object']).columns:
        df_processed[col] = df_processed[col].astype('category')
    df_processed = pd.get_dummies(df_processed, drop_first=True)
    return df_processed

def postprocess_data(synthetic_df, original_df):
    df_final = synthetic_df.copy()
    for col in original_df.columns:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(original_df[col].dtype, errors='ignore')
    return df_final

def align_columns(df_to_align, reference_columns):
    aligned_df = pd.DataFrame(columns=reference_columns, index=df_to_align.index)
    common_cols = [col for col in df_to_align.columns if col in reference_columns]
    aligned_df[common_cols] = df_to_align[common_cols]
    return aligned_df.fillna(0)

def run_ml_utility_test(real_df_raw, synthetic_df_final, target_col_name, task_mode):
    if task_mode != 'unsupervised' and (target_col_name not in real_df_raw.columns or synthetic_df_final.empty): return {}
    
    metrics = {}
    if task_mode == 'classification' or task_mode == 'regression':
        X_real, y_real = real_df_raw.drop(columns=[target_col_name]), real_df_raw[target_col_name]
        X_synth, y_synth = synthetic_df_final.drop(columns=[target_col_name]), synthetic_df_final[target_col_name]
        X_train_processed, X_test_processed = prepare_data(X_synth), prepare_data(X_real)
        X_test_aligned = align_columns(X_test_processed, X_train_processed.columns)
        
        if task_mode == 'classification':
            model = RandomForestClassifier(random_state=42, n_estimators=50).fit(X_train_processed, y_synth)
            preds = model.predict(X_test_aligned)
            metrics['accuracy'] = accuracy_score(y_real, preds)
            metrics['f1_score'] = f1_score(y_real, preds, average='weighted')
        else: # regression
            model = RandomForestRegressor(random_state=42, n_estimators=50).fit(X_train_processed, y_synth)
            preds = model.predict(X_test_aligned)
            metrics['r2_score'] = r2_score(y_real, preds)
            metrics['mae'] = mean_absolute_error(y_real, preds)
    else: # unsupervised
        real_processed = prepare_data(real_df_raw)
        synth_processed = prepare_data(synthetic_df_final)
        synth_aligned = align_columns(synth_processed, real_processed.columns)
        
        n_clusters = 5 # A reasonable default
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(real_processed)
        real_labels = kmeans.labels_
        synth_labels = kmeans.predict(synth_aligned)
        metrics['Adjusted Rand Score'] = adjusted_rand_score(real_labels, synth_labels)

    return {k: round(v, 4) for k, v in metrics.items()}

def get_pca_data(real_df_processed, synthetic_df_processed):
    if real_df_processed.empty or synthetic_df_processed.empty: return {}
    synth_aligned = align_columns(synthetic_df_processed, real_df_processed.columns)
    scaler = StandardScaler().fit(real_df_processed)
    real_scaled, synth_scaled = scaler.transform(real_df_processed), scaler.transform(synth_aligned)
    pca = PCA(n_components=2).fit(real_scaled)
    real_pca, synth_pca = pca.transform(real_scaled), pca.transform(synth_scaled)
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    return {
        'real': pd.DataFrame(real_pca, columns=['x', 'y']).to_dict('records'),
        'synthetic': pd.DataFrame(synth_pca, columns=['x', 'y']).to_dict('records'),
        'variance': round(explained_variance, 2)
    }

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
            df = pd.read_csv(file, low_memory=False)
            column_info = analyze_columns(df)
            num_rows, num_cols = df.shape
            is_big_data = (num_rows * num_cols) > app.config['BIG_DATA_CELL_THRESHOLD']
            response_data = {
                "categorical": column_info['categorical'],
                "numerical": column_info['numerical'],
                "is_big_data": is_big_data,
                "row_count": num_rows,
                "col_count": num_cols,
                "default_sample_size": app.config['DEFAULT_SAMPLE_SIZE']
            }
            return jsonify(response_data)
        except Exception as e:
            app.logger.error(f"Error in /info: {e}")
            return jsonify({"error": "Could not process CSV file."}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/generate', methods=['POST'])
def generate_synthetic_data():
    try:
        start_time = time.time()
        file = request.files['dataset']
        
        original_filename = secure_filename(file.filename)
        real_data_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.seek(0); file.save(real_data_path)
        
        class_col_name = request.form.get('class_col', None)
        if not class_col_name or class_col_name == "none": class_col_name = None
        
        real_df_raw = pd.read_csv(real_data_path, low_memory=False)
        
        original_row_count, original_col_count = real_df_raw.shape
        task_mode = 'unsupervised'
        if class_col_name:
            y_raw = real_df_raw[class_col_name]
            task_mode = 'classification' if y_raw.nunique() <= 25 else 'regression'
        is_big_data_mode = 'big_data_mode' in request.form
        data_to_process = real_df_raw
        if is_big_data_mode:
            sample_size_str = request.form.get('sample_size')
            sample_size = int(sample_size_str) if sample_size_str and sample_size_str.isdigit() else app.config['DEFAULT_SAMPLE_SIZE']
            if sample_size < original_row_count:
                if task_mode == 'classification' and class_col_name:
                    data_to_process = real_df_raw.groupby(class_col_name, group_keys=False).apply(lambda x: x.sample(n=min(len(x), int(np.ceil(sample_size * len(x) / original_row_count))), random_state=42)).sample(frac=1, random_state=42)
                else:
                    data_to_process = real_df_raw.sample(n=sample_size, random_state=42)
        
        real_df_processed = prepare_data(data_to_process)
        max_k = len(data_to_process) - 1
        if task_mode == 'classification':
            min_class_size = data_to_process[class_col_name].value_counts().min()
            max_k = min(max_k, min_class_size)
        k_options = [k for k in [3, 5, 7] if k < max_k]
        best_k = k_options[0] if k_options else 3
        n_obs, n_epochs = int(request.form['n_obs']), int(request.form['n_epochs'])
        X, y = real_df_processed.copy(), None
        if class_col_name: y = X.pop(class_col_name)
        model = kNNMTD(n_obs=n_obs, k=best_k, n_epochs=n_epochs)
        
        pcd_over_time, ml_metrics_over_time = [], {}
        initial_metrics, final_metrics = {}, {}
        all_batches = list(model.fit_generate(X, y))
        if not all_batches: return render_template('error.html', error_message="Failed to generate any synthetic data.")

        for batch_processed in all_batches:
            batch_final = postprocess_data(batch_processed, data_to_process)
            pcd_over_time.append(PCD(data_to_process, batch_final))
            if task_mode != 'unsupervised':
                metrics = run_ml_utility_test(data_to_process, batch_final, class_col_name, task_mode)
                for key, value in metrics.items():
                    ml_metrics_over_time.setdefault(key, []).append(value)
        
        final_synthetic_df = postprocess_data(all_batches[-1], data_to_process)
        
        if pcd_over_time:
            initial_metrics["PCD"] = pcd_over_time[0]
            final_metrics["PCD"] = pcd_over_time[-1]
        for key, values in ml_metrics_over_time.items():
            if values:
                initial_metrics[key] = values[0]
                final_metrics[key] = values[-1]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        synthetic_filename = f"synthetic_{timestamp}_{original_filename}"
        final_synthetic_df.to_csv(os.path.join(app.config['GENERATED_FOLDER'], synthetic_filename), index=False)
        
        if 'augment' in request.form.get('generation_mode', ''):
            final_df_to_save = pd.concat([real_df_raw, final_synthetic_df], ignore_index=True)
            final_df_to_save.to_csv(os.path.join(app.config['GENERATED_FOLDER'], synthetic_filename), index=False)
        
        total_runtime = round(time.time() - start_time, 2)

        return render_template('results.html', 
                               pcd_scores=pcd_over_time,
                               ml_metrics_over_time=ml_metrics_over_time,
                               generated_filename=synthetic_filename,
                               original_real_filename=original_filename,
                               initial_metrics=initial_metrics,
                               final_metrics=final_metrics,
                               pipeline_stages=[
                                   ("Data Preparation", "Cleaned & encoded data"),
                                   ("Scaling Strategy", f"Processed {len(data_to_process)} rows" if is_big_data_mode else "Full dataset processed"),
                                   ("Generation", f"Created {n_obs} new samples using k={best_k}"),
                                   ("Evaluation", "Metrics calculated over epochs")
                               ],
                               input_options={
                                   "Dataset": original_filename,
                                   "Original Data": f"{original_row_count} rows, {original_col_count} columns",
                                   "Final Synthetic Data": f"{len(final_synthetic_df)} rows, {len(final_synthetic_df.columns)} columns",
                                   "Task": task_mode.capitalize(),
                                   "Target": class_col_name if class_col_name else "None",
                                   "Total Runtime": f"{total_runtime} seconds"
                               })
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {traceback.format_exc()}")
        return render_template('error.html', error_message="An internal error occurred. Details have been logged.")

@app.route('/analyze')
def analyze():
    real_filename = request.args.get('real')
    synth_filename = request.args.get('synth')

    if not real_filename or not synth_filename:
        return "Error: Missing data file parameters.", 400

    real_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], real_filename))
    
    all_columns = sorted(real_df.columns.tolist())

    return render_template('analysis.html', 
                           columns=all_columns, 
                           real_filename=real_filename,
                           synth_filename=synth_filename)

@app.route('/get_analysis_data', methods=['POST'])
def get_analysis_data():
    data = request.json
    real_filename = data.get('real_filename')
    synth_filename = data.get('synth_filename')
    analysis_type = data.get('analysis_type')
    selected_columns = data.get('columns')

    real_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], real_filename))
    synth_df = pd.read_csv(os.path.join(app.config['GENERATED_FOLDER'], synth_filename))

    if analysis_type == 'distribution':
        col = selected_columns[0]
        col_type = 'categorical' if pd.api.types.is_object_dtype(real_df[col]) or real_df[col].nunique() < 20 else 'numerical'
        
        if col_type == 'categorical':
            real_counts = real_df[col].value_counts(normalize=True).to_dict()
            synth_counts = synth_df[col].value_counts(normalize=True).to_dict()
            return jsonify({ "type": "categorical", "real": real_counts, "synth": synth_counts })
        else:
            real_stats = { 'Mean': round(real_df[col].mean(), 2), 'Std Dev': round(real_df[col].std(), 2), 'Min': round(real_df[col].min(), 2), 'Max': round(real_df[col].max(), 2) }
            synth_stats = { 'Mean': round(synth_df[col].mean(), 2), 'Std Dev': round(synth_df[col].std(), 2), 'Min': round(synth_df[col].min(), 2), 'Max': round(synth_df[col].max(), 2) }
            return jsonify({ "type": "numerical", "real": real_df[col].dropna().tolist(), "synth": synth_df[col].dropna().tolist(), "stats": {"real": real_stats, "synth": synth_stats} })

    elif analysis_type == 'correlation':
        real_corr = associations(real_df[selected_columns], compute_only=True)['corr']
        synth_corr = associations(synth_df[selected_columns], compute_only=True)['corr']
        return jsonify({ 'labels': real_corr.columns.tolist(), 'real_matrix': real_corr.values.tolist(), 'synth_matrix': synth_corr.values.tolist() })

    elif analysis_type == 'pca':
        real_df_processed, synth_df_processed = prepare_data(real_df), prepare_data(synth_df)
        pca_data = get_pca_data(real_df_processed, synth_df_processed)
        return jsonify(pca_data)

    elif analysis_type == 'ml_efficacy':
        target_col = data.get('target_col', None)
        task = 'unsupervised'
        if target_col:
            task = 'classification' if real_df[target_col].nunique() <= 25 else 'regression'
        
        tstr_metrics = run_ml_utility_test(real_df, synth_df, target_col, task)
        trts_metrics = run_ml_utility_test(synth_df, real_df, target_col, task)
        return jsonify({'tstr': tstr_metrics, 'trts': trts_metrics, 'task': task})

    elif analysis_type == 'privacy':
        real_processed = prepare_data(real_df[selected_columns])
        synth_processed = prepare_data(synth_df[selected_columns])
        
        real_processed['is_synthetic'] = 0
        synth_processed['is_synthetic'] = 1
        
        combined = pd.concat([real_processed, synth_processed], ignore_index=True)
        combined_aligned = align_columns(combined, combined.columns)
        
        X = combined_aligned.drop(columns=['is_synthetic'])
        y = combined_aligned['is_synthetic']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        return jsonify({'auc': round(auc, 4)})

    return jsonify({"error": "Invalid analysis type"})


@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    folder_path = app.config.get(folder.upper() + '_FOLDER')
    if not folder_path:
        return "Invalid folder specified", 404
    return send_from_directory(folder_path, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)