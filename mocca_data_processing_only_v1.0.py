import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
from mocca2 import Chromatogram, MoccaDataset, ProcessingSettings
from mocca2.classes import Data2D
import matplotlib.pyplot as plt
import re
import pickle
from datetime import datetime
import seaborn as sns
import numpy as np
import rainbow as rb

# ---------- App Helper Functions ----------
def generate_well_positions(tray_size,n):
    if tray_size == "48":
        rows = ['A','B','C','D','E','F']
        cols = list(range(1, 9))
        positions = [f"{row}{col}" for row in rows for col in cols]
    elif tray_size == "96":
        rows = ['A','B','C','D','E','F','G','H']
        cols = list(range(1, 13))
        positions = [f"{row}{col}" for row in rows for col in cols]
    return positions[:n]

# ---------- Streamlit App ----------
st.title("MOCCA - Data Processing")

# Step 1: Upload metadata and zip
st.subheader("Step 1: Upload Files")
metadata_file = st.file_uploader("Upload Metadata CSV", type=["csv"])
raw_zip_file = st.file_uploader("Upload a ZIP containing .raw folders", type=["zip"])

# Step 2: Settings
if st.toggle("Use custom processing settings"):
    st.subheader("Step 2: Optional Settings")
    min_elution_time = st.number_input("Minimum Elution Time", value=0.30, step=0.01, key="3")
    max_elution_time = st.number_input("Maximum Elution Time", value=1.65, step=0.01, key="4")
    min_wavelength = st.number_input("Minimum Wavelength", value=230, step=1, key="5")
    max_wavelength = st.number_input("Maximum Wavelength", value=400, step=1, key="5.5")
    min_peak_height_batch = st.number_input("Minimum Peak Height", value=0.01, step=0.0001, format="%.4f", key="7")
    min_rel_peak_height_batch = st.number_input("Minimum Peak Relative Height", value=0.001, step=0.00001, format="%.5f", key="8")
    r2_input_batch = st.number_input("Peak Deconvolution Threshold", value=0.998, step=0.0001, format="%.4f",key="10")
    min_spectrum_corr = st.number_input("Minimum specturm correlation", value=0.900, step=0.0001, format="%.4f",key="11")
    min_rel_integral = st.number_input("Minimum relative peak integral", value=0.01000, step=0.00001, format="%.4f",key="12")

else:
    min_elution_time = 0.3
    max_elution_time = 1.65
    min_wavelength = 230
    max_wavelength = 400
    min_peak_height_batch = -np.inf
    min_rel_peak_height_batch = 0.001
    r2_input_batch = 0.998
    min_spectrum_corr = 0.9
    min_rel_integral = 0.001

# Step 3: Run processing
st.session_state["proc_finished"] = False
if st.button("Run Mocca2 Processing") and metadata_file and raw_zip_file:

    with st.spinner("Processing LC data..."):
        st.write("Loading metdata...")
        metadata = pd.read_csv(metadata_file)

        st.write("Finding chromatograms...")
        # Extract zip to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "raw_data.zip")
            with open(zip_path, "wb") as f:
                f.write(raw_zip_file.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Recursively find all .raw folders (case-insensitive)
            masslynx_folders = True
            extracted_dirs = {}
            for root, dirs, _ in os.walk(temp_dir):
                for d in dirs:
                    if d.lower().endswith(".raw"):
                        extracted_dirs[d.lower()] = os.path.join(root, d)

            dataset = MoccaDataset()
            blankChrom = None
            
            st.write("Loading chromatograms...")
            for idx, row in metadata.iterrows():

                sample_metadata = row
                sample_name = str(sample_metadata["FILE_NAME"]).strip()
                folder_key = (sample_name + ".raw").lower()
                raw_path = extracted_dirs.get(folder_key)
                if raw_path is None:
                    st.warning(f"Could not find .raw folder for sample: {sample_name}")
                    continue

                chromData = raw_path

                if sample_metadata["Is blank"]:
                    blankChrom = chromData
                    continue
                
                chrom = Chromatogram(chromData, blankChrom, name=sample_name)                                

                if sample_metadata["Is marker"]:
                    dataset.add_chromatogram(
                        chrom,
                        reference_for_compound=sample_metadata["Compound"],
                        compound_concentration=sample_metadata["Compound Conc."],
                        istd_concentration=0
                    )
                else:
                    dataset.add_chromatogram(
                        chrom,
                        istd_concentration=0
                    )
                
        settings_kwargs = dict(
            baseline_model="flatfit",
            peak_model="Bemg",
            min_elution_time=min_elution_time,
            max_elution_time=max_elution_time,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            explained_threshold=r2_input_batch,
            min_spectrum_correl=min_spectrum_corr,
            min_rel_integral=min_rel_integral,
            min_prominence=min_peak_height_batch,
            min_rel_prominence=min_rel_peak_height_batch,
            split_threshold=None,
            max_peak_comps=5
        )
    
        settings = ProcessingSettings(**settings_kwargs)

        try:
            dataset.process_all(settings, verbose=True, cores=os.cpu_count()-1)
        except:
            settings_kwargs = dict(
                baseline_model="flatfit",
                peak_model="Bemg",
                min_elution_time=min_elution_time,
                max_elution_time=max_elution_time,
                min_wavelength=min_wavelength,
                max_wavelength=max_wavelength,
                explained_threshold=r2_input_batch,
                min_spectrum_correl=min_spectrum_corr,
                min_rel_integral=min_rel_integral,
                min_prominence=min_peak_height_batch,
                min_rel_prominence=min_rel_peak_height_batch*10,
                split_threshold=None,
                max_peak_comps=5
            )
            settings = ProcessingSettings(**settings_kwargs)
            dataset.process_all(settings, verbose=True, cores=os.cpu_count()-1)
        
            st.session_state["dataset"] = dataset
            st.success("Processing complete!")
            st.session_state["proc_finished"] = True

        savefilebasename = metadata_file.name.split("_sample_list.csv")[0]

if st.session_state.proc_finished:

    dataset = st.session_state["dataset"]
    st.download_button(
    "Download Full Dataset",
    data=pickle.dumps(dataset.to_dict()),
    file_name=f"{savefilebasename}_{datetime.now()}_MOCCA_dataset.pkl",
    mime='application/octet-stream'
)

    st.download_button(
    "Download Peak Areas",
    data=pickle.dumps({"peak areas": dataset.get_integrals(), "peak concentrations": dataset.get_concentrations()}),
    file_name=f"{savefilebasename}_{datetime.now()}_peak_info_only.pkl",
    mime='application/octet-stream'
)