from __future__ import annotations
import json, os, tempfile, uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from core.model import choose_device, load_model
from core.preprocessing import PreprocessConfig, display_normalize, load_case
from core.inference import run_case_inference
from core.severity import load_calibration, assign_severity_score, severity_label, confidence_category
from core.recommendation import generate_decision_support
from core.storage import history_dataframe, init_db, persist_case

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / "models" / "best_model3_unetplusplus_brats2021.pt"
THRESHOLD_FILE = BASE / "config" / "selected_threshold.json"
MODEL_CARD_FILE = BASE / "config" / "model3_model_card.json"
CALIBRATION_FILE = BASE / "config" / "severity_calibration.json"
SAMPLE_DIR = BASE / "sample_case"

st.set_page_config(page_title="BT-MRI DSS", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.hero {padding: 1rem 1.2rem; border: 1px solid #dfe6ee; border-radius: 14px; background: #f8fafc; margin-bottom: 1rem;}
.small-note {font-size: 0.88rem; color: #59636e;}
</style>
""", unsafe_allow_html=True)

model_card = json.loads(MODEL_CARD_FILE.read_text())
threshold_info = json.loads(THRESHOLD_FILE.read_text())
threshold = float(threshold_info["selected_threshold"])
calibration = load_calibration(CALIBRATION_FILE)

if "last_analysis" not in st.session_state: st.session_state.last_analysis = None
if "results_root" not in st.session_state: st.session_state.results_root = str(Path.home() / "BT_MRI_DSS_RESULTS")

with st.sidebar:
    st.title("BT-MRI DSS")
    page = st.radio("Navigation", ["Dashboard", "New Analysis", "Case Results", "History", "System Information"])
    st.divider()
    st.session_state.results_root = st.text_input("Results folder", value=st.session_state.results_root)
    device_pref = st.selectbox("Compute device", ["auto", "cuda", "cpu"], index=0)
    st.caption(f"Segmentation threshold: {threshold:.2f}")
    if calibration:
        st.success("Severity calibration loaded")
    else:
        st.warning("Severity calibration required")
    st.caption("Research decision-support prototype; not a medical diagnosis.")

results_root = Path(st.session_state.results_root).expanduser()
init_db(results_root)

@st.cache_resource(show_spinner=False)
def get_model(device_name: str):
    dev = choose_device(device_name)
    model, ckpt = load_model(CHECKPOINT, dev)
    return model, ckpt, dev


def checkpoint_config(ckpt):
    c = ckpt.get("config", {})
    return PreprocessConfig(img_size=int(c.get("img_size",192)), percentile_low=float(c.get("percentile_low",1.0)), percentile_high=float(c.get("percentile_high",99.5)), zscore_clip=float(c.get("zscore_clip",5.0)), use_brain_crop=bool(c.get("use_brain_crop",True)), crop_padding=int(c.get("crop_padding",8)))


def write_uploaded(uploaded, target: Path):
    target.write_bytes(uploaded.getbuffer()); return target


def severity_enrich(result: dict):
    result = dict(result)
    if not calibration:
        result.update({"severity_score":None,"severity_label":"Calibration required","confidence_category":"Not assigned","urgency_category":"Not assigned","safety_modifiers":"Severity calibration required before recommendation generation."})
        return result
    score = assign_severity_score(result, calibration)
    result["severity_score"] = score
    result["severity_label"] = severity_label(score)
    result["confidence_category"] = confidence_category(result)
    result.update(generate_decision_support(result))
    return result


def render_case(result, aux, volumes, artifacts=None):
    sev = result.get("severity_score")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Tumor burden (B)", f"{result['predicted_tumor_burden']:.4f}")
    m2.metric("Affected slices (A)", f"{result['affected_slice_ratio']:.4f}")
    m3.metric("Max slice ratio (M)", f"{result['max_slice_tumor_ratio']:.4f}")
    m4.metric("Severity", "Pending" if sev is None else f"{sev}/5")
    m5.metric("Confidence", result.get("confidence_category","N/A"))

    if sev is None:
        st.warning("Segmentation is complete, but severity/recommendation output is withheld until U-Net++ validation calibration is generated.")
    else:
        st.info(f"**{result['severity_label']}**  ·  **{result['urgency_category']}**")
        st.write(result.get("structured_summary",""))
        st.write("**Recommended next step:**", result.get("recommended_next_step",""))
        st.write("**Safety modifiers:**", result.get("safety_modifiers",""))

    tabs = st.tabs(["MRI Viewer", "Tumor Profile", "Evaluation", "3D View", "Report"])
    with tabs[0]:
        idx = st.slider("Valid axial slice", 0, len(aux["z_indices"])-1, int(np.argmax(aux["pred_pixels_per_slice"])) if np.max(aux["pred_pixels_per_slice"])>0 else len(aux["z_indices"])//2)
        z = int(aux["z_indices"][idx])
        c1,c2,c3,c4 = st.columns(4)
        for col,key,label in zip((c1,c2,c3,c4),("flair","t1","t1ce","t2"),("FLAIR","T1","T1CE","T2")):
            col.image(display_normalize(volumes[key][:,:,z]), caption=f"{label} · z={z}", clamp=True, use_container_width=True)
        c5,c6 = st.columns(2)
        c5.image(aux["pred_volume_original"][:,:,z]*255, caption="Predicted whole-tumor mask", clamp=True, use_container_width=True)
        fig=go.Figure(); fig.add_trace(go.Heatmap(z=aux["prob_volume_original"][:,:,z].T, colorscale="Magma", zmin=0,zmax=1, colorbar=dict(title="P"))); fig.update_layout(title=f"Probability map · z={z}", height=430, margin=dict(l=10,r=10,t=45,b=10)); c6.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        df=pd.DataFrame({"Slice":aux["z_indices"],"Predicted tumor ratio":aux["slice_tumor_ratio"]})
        fig=px.line(df,x="Slice",y="Predicted tumor ratio",markers=True,title="Predicted tumor burden across axial slices"); st.plotly_chart(fig,use_container_width=True)
        measures=pd.DataFrame({"Measure":["Tumor burden B","Affected-slice ratio A","Maximum slice ratio M","Mean confidence C","Uncertainty U"],"Value":[result["predicted_tumor_burden"],result["affected_slice_ratio"],result["max_slice_tumor_ratio"],result["mean_confidence_in_prediction"],result["uncertainty_ratio"]]})
        st.plotly_chart(px.bar(measures,x="Measure",y="Value",title="Case-level decision-support measurements"),use_container_width=True)
    with tabs[2]:
        if "case_dice" not in result:
            st.caption("Ground-truth segmentation was not supplied; evaluation metrics are not available in Research Analysis mode.")
        else:
            metrics={"Dice":result.get("case_dice"),"IoU":result.get("case_iou"),"Precision":result.get("case_precision"),"Sensitivity":result.get("case_recall_sensitivity"),"Specificity":result.get("case_specificity"),"MCC":result.get("case_mcc"),"VS":result.get("case_volume_similarity"),"RVE":result.get("case_relative_volume_error")}
            st.dataframe(pd.DataFrame([metrics]),use_container_width=True,hide_index=True)
            st.plotly_chart(px.bar(x=list(metrics.keys()),y=list(metrics.values()),labels={"x":"Metric","y":"Value"},title="Case-level evaluation metrics"),use_container_width=True)
    with tabs[3]:
        mask=aux["pred_volume_original"]
        if mask.sum()==0:
            st.caption("No predicted tumor voxels available for 3D visualization.")
        else:
            stride=4
            small=mask[::stride,::stride,::stride]
            zz,yy,xx=np.mgrid[0:small.shape[0],0:small.shape[1],0:small.shape[2]]
            fig=go.Figure(go.Isosurface(x=xx.flatten(),y=yy.flatten(),z=zz.flatten(),value=small.flatten(),isomin=0.5,isomax=1.0,surface_count=1,caps=dict(x_show=False,y_show=False,z_show=False)))
            fig.update_layout(title="Downsampled 3D predicted whole-tumor mask",height=550,scene=dict(xaxis_title="X",yaxis_title="Y",zaxis_title="Z")); st.plotly_chart(fig,use_container_width=True)
    with tabs[4]:
        if artifacts:
            st.success(f"Saved to: {artifacts['case_dir']}")
            st.download_button("Download PDF report", Path(artifacts["report"]).read_bytes(), file_name=f"{result['analysis_id']}_report.pdf", mime="application/pdf")
            st.download_button("Download complete analysis bundle", Path(artifacts["zip"]).read_bytes(), file_name=f"{result['analysis_id']}_bundle.zip", mime="application/zip")
        st.code(json.dumps({k:v for k,v in result.items() if not isinstance(v,(np.ndarray,torch.Tensor))},indent=2,default=str),language="json")


if page == "Dashboard":
    st.markdown('<div class="hero"><h2>Brain Tumor MRI Decision-Support Dashboard</h2><div class="small-note">Archived analyses generated by the selected 2D U-Net++ model.</div></div>', unsafe_allow_html=True)
    hist=history_dataframe(results_root)
    a,b,c,d=st.columns(4)
    a.metric("Analyses",len(hist))
    b.metric("Mean tumor burden",f"{hist['predicted_tumor_burden'].mean():.4f}" if len(hist) else "-")
    c.metric("Mean uncertainty",f"{hist['uncertainty_ratio'].mean():.4f}" if len(hist) else "-")
    d.metric("Safety review cases",int(((hist.get('confidence_category',pd.Series(dtype=str))=='Low confidence')).sum()) if len(hist) else 0)
    if hist.empty:
        st.info("No archived analyses yet. Use New Analysis to process a case.")
    else:
        c1,c2=st.columns(2)
        sev=hist.dropna(subset=["severity_score"]).groupby("severity_score").size().reset_index(name="Cases")
        if len(sev): c1.plotly_chart(px.bar(sev,x="severity_score",y="Cases",title="Severity-score distribution"),use_container_width=True)
        urg=hist["urgency_category"].dropna().value_counts().reset_index(); urg.columns=["Category","Cases"]
        if len(urg): c2.plotly_chart(px.bar(urg,x="Category",y="Cases",title="Recommendation distribution"),use_container_width=True)
        st.dataframe(hist.head(20),use_container_width=True,hide_index=True)

elif page == "New Analysis":
    st.header("New MRI Analysis")
    st.caption("Input must be BraTS-compatible, co-registered 3D NIfTI volumes. The application does not perform registration or clinical image-quality grading.")
    mode=st.radio("Analysis mode",["Research Analysis","Evaluation Mode"],horizontal=True)
    case_id=st.text_input("Case ID",value="Case_001")
    use_sample=st.checkbox("Use bundled BraTS2021 sample case")
    uploads={}
    if not use_sample:
        cols=st.columns(4)
        uploads["flair"]=cols[0].file_uploader("FLAIR",type=["nii","gz"],key="flair")
        uploads["t1"]=cols[1].file_uploader("T1",type=["nii","gz"],key="t1")
        uploads["t1ce"]=cols[2].file_uploader("T1CE",type=["nii","gz"],key="t1ce")
        uploads["t2"]=cols[3].file_uploader("T2",type=["nii","gz"],key="t2")
        if mode=="Evaluation Mode": uploads["seg"]=st.file_uploader("Ground-truth segmentation (optional outside evaluation)",type=["nii","gz"],key="seg")
    if st.button("Process MRI case",type="primary",use_container_width=True):
        try:
            with st.status("Processing MRI case...",expanded=True) as status:
                status.write("Preparing inputs")
                tmp=Path(tempfile.mkdtemp(prefix="bt_mri_dss_"))
                if use_sample:
                    case_id="BraTS2021_00000"
                    paths={k:SAMPLE_DIR/f"BraTS2021_00000_{k}.nii.gz" for k in ("flair","t1","t1ce","t2")}
                    if mode=="Evaluation Mode": paths["seg"]=SAMPLE_DIR/"BraTS2021_00000_seg.nii.gz"
                else:
                    required=["flair","t1","t1ce","t2"] + (["seg"] if mode=="Evaluation Mode" else [])
                    if any(uploads.get(k) is None for k in required): raise ValueError("Please supply all required MRI files for the selected mode.")
                    paths={k:write_uploaded(uploads[k],tmp/f"{case_id}_{k}.nii.gz") for k in required}
                volumes,ref=load_case(paths,require_seg=(mode=="Evaluation Mode"))
                status.write("Loading the selected U-Net++ checkpoint")
                model,ckpt,device=get_model(device_pref)
                pcfg=checkpoint_config(ckpt)
                status.write(f"Running whole-tumor inference on {device}")
                result,aux=run_case_inference(model,volumes,device,threshold,pcfg,batch_size=16,case_id=case_id,include_gt=(mode=="Evaluation Mode"))
                result=severity_enrich(result)
                now=datetime.now(); result["analysis_id"]=f"{case_id}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"; result["timestamp"]=now.isoformat(timespec="seconds"); result["analysis_mode"]=mode
                status.write("Saving result archive and report")
                artifacts=persist_case(result,aux,volumes,ref,results_root,model_card,CHECKPOINT)
                st.session_state.last_analysis={"result":result,"aux":aux,"volumes":volumes,"artifacts":artifacts}
                status.update(label="Analysis completed",state="complete")
            render_case(result,aux,volumes,artifacts)
        except Exception as e:
            st.exception(e)

elif page == "Case Results":
    st.header("Latest Case Results")
    if st.session_state.last_analysis is None:
        st.info("No case is loaded in this session. Process a new case or open a saved report from History.")
    else:
        x=st.session_state.last_analysis; render_case(x["result"],x["aux"],x["volumes"],x["artifacts"])

elif page == "History":
    st.header("Analysis History")
    hist=history_dataframe(results_root)
    if hist.empty:
        st.info("No saved analyses.")
    else:
        search=st.text_input("Search case or analysis ID")
        view=hist.copy()
        if search:
            mask=view["case_id"].astype(str).str.contains(search,case=False,na=False)|view["analysis_id"].astype(str).str.contains(search,case=False,na=False); view=view[mask]
        st.dataframe(view,use_container_width=True,hide_index=True)
        st.download_button("Export history CSV",view.to_csv(index=False).encode(),file_name="bt_mri_dss_history.csv",mime="text/csv")
        choice=st.selectbox("Select saved analysis",view["analysis_id"].tolist()) if len(view) else None
        if choice:
            row=view[view["analysis_id"]==choice].iloc[0]; report=Path(row["report_path"]); bundle=Path(row["case_dir"])/"analysis_bundle.zip"
            if report.exists(): st.download_button("Download selected PDF report",report.read_bytes(),file_name=f"{choice}_report.pdf",mime="application/pdf")
            if bundle.exists(): st.download_button("Download selected analysis bundle",bundle.read_bytes(),file_name=f"{choice}_bundle.zip",mime="application/zip")

else:
    st.header("System Information")
    st.subheader("Selected segmentation model")
    st.json(model_card)
    st.subheader("Methodological alignment")
    st.markdown(f"""
- **Task:** binary whole-tumor segmentation (BraTS labels 1, 2 and 4).
- **Input modalities:** FLAIR, T1, T1CE and T2.
- **Model:** Model 3 - 2D U-Net++.
- **Input size:** 192 × 192 axial slices.
- **Operating threshold:** {threshold:.2f}, selected by validation Dice.
- **Severity:** derived from predicted tumor burden (B), affected-slice ratio (A), and maximum slice tumor ratio (M).
- **Recommendation:** deterministic rule-guided output based on severity, confidence, uncertainty and defined safety modifiers.
""")
    if calibration: st.json(calibration)
    else:
        st.warning("Final U-Net++ severity calibration has not yet been generated. Run calibrate_severity.py against the BraTS2021 validation cases, then restart the app.")
        st.code("python calibrate_severity.py --data-root /path/to/BraTS2021_Training_Data",language="bash")
    st.error("Research-use only. This prototype does not provide a medical diagnosis, pathological grade, treatment prescription, or replacement for specialist interpretation.")
