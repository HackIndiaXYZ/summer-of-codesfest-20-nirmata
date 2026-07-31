"""
MRI QC Dashboard — Streamlit
Uses v3 pipeline (no skull stripping) + XGBoost classifier + artifact detection.
Run: streamlit run app.py
"""

import io
import os
import json
import tempfile
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

from pipeline import extract_scan, find_nifti_files, FEATURE_COLUMNS

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MRI QC Dashboard",
    page_icon="MRI",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = ['fail', 'pass', 'review']
STATUS_COLORS = {"PASS": "#2ecc71", "REVIEW": "#f39c12", "FAIL": "#e74c3c"}
ARTIFACT_ICONS = {'none': 'OK', 'mild': '~', 'moderate': '!', 'severe': 'X'}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgboost_mri_qc_model.json")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "mriqc_combined_dataset.json")


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_reference_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def classify_scan(features, model):
    feature_vector = np.array([[features[col] for col in FEATURE_COLUMNS]])
    pred_class = model.predict(feature_vector)[0]
    pred_proba = model.predict_proba(feature_vector)[0]
    return {
        'prediction': CLASS_NAMES[pred_class],
        'confidence': float(np.max(pred_proba)),
        'probabilities': {
            'fail': float(pred_proba[0]),
            'pass': float(pred_proba[1]),
            'review': float(pred_proba[2])
        }
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("MRI QC Dashboard")
st.sidebar.markdown("**Team NIRMATA**")

page = st.sidebar.radio(
    "Navigate",
    ["Batch Scorecard", "Analytics", "Scan Drill-down"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Input Scans")

input_method = st.sidebar.radio(
    "Input method",
    ["Upload NIfTI file(s)", "Local folder path", "Upload ZIP"],
)

# --------------------------------------------------------------------------
# Process scans
# --------------------------------------------------------------------------
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
    st.session_state.scan_data = {}

model = load_model()
ref_df = load_reference_dataset()

if model is None:
    st.sidebar.error(
        f"Model file not found at: {MODEL_PATH}\n\n"
        "Place your xgboost_mri_qc_model.json in the dashboard folder."
    )


def process_files(file_paths, progress_bar=None):
    results = {}
    for idx, fp in enumerate(file_paths):
        scan_name = os.path.basename(fp).replace(".nii.gz", "").replace(".nii", "")
        result = extract_scan(fp)
        if result is not None:
            qc = classify_scan(result['features'], model)
            results[scan_name] = {
                **result['features'],
                'prediction': qc['prediction'],
                'confidence': qc['confidence'],
                'probabilities': qc['probabilities'],
                'artifacts': result['artifacts'],
                'meta': result['meta'],
                'filepath': fp,
            }
            st.session_state.scan_data[scan_name] = {
                'data': result['data'],
                'head_mask': result['head_mask'],
                'brain_mask': result['brain_mask'],
            }
        if progress_bar:
            progress_bar.progress((idx + 1) / len(file_paths))
    return results


if input_method == "Upload NIfTI file(s)":
    uploaded_files = st.sidebar.file_uploader(
        "Upload .nii or .nii.gz files",
        type=["nii", "gz"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.sidebar.button("Process Scans"):
        with st.spinner("Processing scans..."):
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for uf in uploaded_files:
                path = os.path.join(temp_dir, uf.name)
                with open(path, 'wb') as f:
                    f.write(uf.getbuffer())
                file_paths.append(path)
            progress = st.sidebar.progress(0)
            st.session_state.scan_results = process_files(file_paths, progress)
            st.sidebar.success(f"Processed {len(st.session_state.scan_results)} scans")

elif input_method == "Local folder path":
    folder_path = st.sidebar.text_input(
        "Folder path",
        placeholder="C:\\path\\to\\nifti_files",
    )
    if folder_path and st.sidebar.button("Scan Folder"):
        if os.path.isdir(folder_path):
            nifti_files = find_nifti_files(folder_path)
            if nifti_files:
                with st.spinner(f"Processing {len(nifti_files)} files..."):
                    progress = st.sidebar.progress(0)
                    st.session_state.scan_results = process_files(nifti_files, progress)
                    st.sidebar.success(f"Processed {len(st.session_state.scan_results)} scans")
            else:
                st.sidebar.error("No NIfTI files found in that folder.")
        else:
            st.sidebar.error("Invalid folder path.")

elif input_method == "Upload ZIP":
    zip_upload = st.sidebar.file_uploader("Upload ZIP of NIfTI files", type=["zip"])
    if zip_upload and st.sidebar.button("Process ZIP"):
        with st.spinner("Extracting and processing..."):
            temp_dir = tempfile.mkdtemp()
            zip_bytes = zip_upload.getvalue()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(temp_dir)
            nifti_files = find_nifti_files(temp_dir)
            if nifti_files:
                progress = st.sidebar.progress(0)
                st.session_state.scan_results = process_files(nifti_files, progress)
                st.sidebar.success(f"Processed {len(st.session_state.scan_results)} scans")
            else:
                st.sidebar.error("No NIfTI files found in the ZIP.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Scans loaded: {len(st.session_state.scan_results)}")

results = st.session_state.scan_results


# ==========================================================================
# PAGE 1 — Batch Scorecard
# ==========================================================================
if page == "Batch Scorecard":
    st.title("Batch QC Scorecard")

    if not results:
        st.info("Upload or select NIfTI files from the sidebar to get started.")
        if ref_df is not None:
            st.markdown("---")
            st.subheader("Reference Dataset Summary")
            from collections import Counter
            labels = Counter(ref_df['quality_label'])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Samples", len(ref_df))
            c2.metric("Pass", labels.get('pass', 0))
            c3.metric("Review", labels.get('review', 0))
            c4.metric("Fail", labels.get('fail', 0))
        st.stop()

    # Build dataframe
    rows = []
    for scan_id, r in results.items():
        row = {
            'Scan_ID': scan_id,
            'QC_Status': r['prediction'].upper(),
            'Confidence': r['confidence'],
            'Contrast': r['meta']['contrast'],
            'Volume_cm3': r['meta']['brain_volume_cm3'],
        }
        for col in FEATURE_COLUMNS:
            row[col] = r[col]
        for art_name, art_info in r['artifacts'].items():
            row[f'artifact_{art_name}'] = art_info['severity']
        rows.append(row)

    df = pd.DataFrame(rows)

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Scans", len(df))
    c2.metric("PASS", int((df.QC_Status == "PASS").sum()))
    c3.metric("REVIEW", int((df.QC_Status == "REVIEW").sum()))
    c4.metric("FAIL", int((df.QC_Status == "FAIL").sum()))
    avg_conf = df['Confidence'].mean()
    c5.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.markdown("---")

    # Styled table
    def highlight_status(val):
        color = STATUS_COLORS.get(val, "")
        if color:
            return f"background-color: {color}; color: white; font-weight: 600;"
        return ""

    display_cols = ['Scan_ID', 'QC_Status', 'Confidence', 'Contrast',
                    'artifact_motion', 'artifact_ringing', 'artifact_noise', 'artifact_blur',
                    'snr_total', 'cnr', 'cjv', 'efc', 'fber']
    display_cols = [c for c in display_cols if c in df.columns]

    styled = (
        df[display_cols]
        .style.map(highlight_status, subset=["QC_Status"])
        .format({"Confidence": "{:.1%}"})
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # Export
    st.markdown("---")
    st.subheader("Export Report")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download QC Report (CSV)",
        data=csv_bytes,
        file_name="mri_qc_report.csv",
        mime="text/csv",
    )


# ==========================================================================
# PAGE 2 — Analytics
# ==========================================================================
elif page == "Analytics":
    st.title("Batch Analytics")

    if not results:
        st.info("No scans processed yet. Use the sidebar to load data.")
        st.stop()

    rows = []
    for scan_id, r in results.items():
        row = {'Scan_ID': scan_id, 'QC_Status': r['prediction'].upper(),
               'Confidence': r['confidence']}
        for col in FEATURE_COLUMNS:
            row[col] = r[col]
        for art_name, art_info in r['artifacts'].items():
            row[f'artifact_{art_name}'] = art_info['severity']
        rows.append(row)
    df = pd.DataFrame(rows)

    tab1, tab2, tab3 = st.tabs(["QC Distribution", "Feature Analysis", "Artifact Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            status_dist = df["QC_Status"].value_counts().reset_index()
            status_dist.columns = ["Status", "Count"]
            fig = px.pie(
                status_dist, names="Status", values="Count", hole=0.45,
                color="Status", color_discrete_map=STATUS_COLORS,
                title="PASS / REVIEW / FAIL",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(
                df, x="Confidence", color="QC_Status",
                color_discrete_map=STATUS_COLORS,
                title="Confidence Distribution",
                nbins=20,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        feat_choice = st.selectbox("Feature", FEATURE_COLUMNS, index=FEATURE_COLUMNS.index('snr_total'))
        fig3 = px.box(
            df, x="QC_Status", y=feat_choice, color="QC_Status",
            color_discrete_map=STATUS_COLORS,
            points="outliers", title=f"{feat_choice} by QC Status",
        )
        st.plotly_chart(fig3, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig4 = px.scatter(
                df, x="cnr", y="cjv", color="QC_Status",
                color_discrete_map=STATUS_COLORS,
                title="CNR vs CJV",
            )
            st.plotly_chart(fig4, use_container_width=True)
        with col2:
            fig5 = px.scatter(
                df, x="efc", y="fwhm_avg", color="QC_Status",
                color_discrete_map=STATUS_COLORS,
                title="EFC vs FWHM (Sharpness)",
            )
            st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        art_cols = ['artifact_motion', 'artifact_ringing', 'artifact_noise', 'artifact_blur']
        art_cols = [c for c in art_cols if c in df.columns]
        if art_cols:
            art_data = []
            for col in art_cols:
                counts = df[col].value_counts()
                for severity, count in counts.items():
                    art_data.append({
                        'Artifact': col.replace('artifact_', '').capitalize(),
                        'Severity': severity,
                        'Count': count
                    })
            art_df = pd.DataFrame(art_data)
            severity_colors = {'none': '#2ecc71', 'mild': '#f1c40f', 'moderate': '#f39c12', 'severe': '#e74c3c'}
            fig6 = px.bar(
                art_df, x="Artifact", y="Count", color="Severity",
                color_discrete_map=severity_colors,
                title="Artifact Severity Distribution",
                barmode="group",
            )
            st.plotly_chart(fig6, use_container_width=True)


# ==========================================================================
# PAGE 3 — Scan Drill-down
# ==========================================================================
elif page == "Scan Drill-down":
    st.title("Scan Drill-down")

    if not results:
        st.info("No scans processed yet. Use the sidebar to load data.")
        st.stop()

    scan_ids = list(results.keys())
    scan_id = st.selectbox("Select scan", scan_ids)
    r = results[scan_id]

    status = r['prediction'].upper()
    color = STATUS_COLORS[status]
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:10px;background-color:{color};color:white;">
        <h3 style="margin:0;">Status: {status}</h3>
        <p style="margin:0;">Confidence: <b>{r['confidence']:.1%}</b>
        &nbsp;|&nbsp; Contrast: <b>{r['meta']['contrast']}</b>
        &nbsp;|&nbsp; Volume: <b>{r['meta']['brain_volume_cm3']:.1f} cm3</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("###")

    # Probabilities + Artifacts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": ['Fail', 'Pass', 'Review'],
            "Probability": [r['probabilities']['fail'], r['probabilities']['pass'], r['probabilities']['review']]
        })
        fig = px.bar(prob_df, x="Class", y="Probability", color="Class",
                     color_discrete_map={'Pass': '#2ecc71', 'Review': '#f39c12', 'Fail': '#e74c3c'},
                     range_y=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Artifacts Detected")
        for art_name, art_info in r['artifacts'].items():
            icon = ARTIFACT_ICONS.get(art_info['severity'], '?')
            sev = art_info['severity']
            sev_color = {'none': 'green', 'mild': 'orange', 'moderate': 'darkorange', 'severe': 'red'}.get(sev, 'gray')
            st.markdown(f"**{art_name.capitalize()}:** :{sev_color}[{sev}] (score: {art_info['score']:.4f})")

    # Feature table
    st.markdown("---")
    st.subheader("IQM Feature Values")
    feat_df = pd.DataFrame([{col: r[col] for col in FEATURE_COLUMNS}])
    st.dataframe(feat_df.T.rename(columns={0: 'Value'}), use_container_width=True)

    # Slice viewer
    st.markdown("---")
    st.subheader("Slice Viewer")
    if scan_id in st.session_state.scan_data:
        volume = st.session_state.scan_data[scan_id]['data']
        n_slices = volume.shape[2]
        z = st.slider("Axial slice", 0, n_slices - 1, n_slices // 2)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(
                volume[:, :, z].T, color_continuous_scale="gray", origin="lower",
            )
            fig.update_layout(title=f"Axial slice {z + 1}/{n_slices}", coloraxis_showscale=False)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            head_mask = st.session_state.scan_data[scan_id]['head_mask']
            overlay = volume[:, :, z].T.copy()
            mask_slice = head_mask[:, :, z].T
            fig2 = px.imshow(
                overlay, color_continuous_scale="gray", origin="lower",
            )
            fig2.update_layout(title="With head mask boundary", coloraxis_showscale=False)
            fig2.update_xaxes(visible=False)
            fig2.update_yaxes(visible=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Volume data not available for slice viewing.")
