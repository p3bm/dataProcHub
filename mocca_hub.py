import streamlit as st
import pandas as pd
import tempfile
import os
import zipfile
import rainbow as rb
from mocca2.classes import Data2D
from mocca2 import Chromatogram, MoccaDataset, ProcessingSettings
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ---------- MassLynx Sample List Helper ----------
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
st.title("HTS Analytical Hub")

with st.expander("Instructions"):
    st.write("""
        1. Generate a sample list for submission of LC samples through MassLynx.
        2. Create a metadata file to pass sample information through to MOCCA processing. Sample names from step 1 can be pulled through here.
        3. If necessary, process single LC samples to assess the peak picking and deconvolution process.
        4. Batch process LC samples to ouput a .csv file of peak areas or concentrations.
        5. Visualise data from a batch of samples processed in step 4. The current data can be pulled through from step 4.
    """)

tab1, tab3, tab4, tab5 = st.tabs([
    "Sample List Generator",
    "Single Sample Analysis",
    "Batch Analysis",
    "Data Visualisation"
    ])

# ----------- Tab 1: MassLynx Sample List Generator -----------
with tab1:
    st.header("📄 MassLynx Sample List Generator")

    num_samples = st.number_input("Number of Samples", min_value=1, max_value=96, value=48)
    base_name = st.text_input("Base Sample Name", value="ELNXXXX-XXX")
    lc_method = st.selectbox("LC Method", options=["Acidic", "Basic"])
    tray_no = st.number_input("Tray No.", 1, 10)
    tray_size = st.selectbox("Tray size", options=["48","96"])
    is_conc = st.number_input("Internal Standard Concentration", min_value=0.0000, step=0.0001, format="%.4f")
    
    # Assign method-specific values
    if lc_method == "Acidic":
        inlet_file = "Method3_S1_C3_TQD_2mins"
        inlet_switch = "Switch_Acidic_C3"
    else:
        inlet_file = "Method7_S1_C3_TQD_2mins"
        inlet_switch = "Switch_Neutral_C3"

    # Generate sample data
    file_names = []
    sample_locations = generate_well_positions(tray_size, num_samples)
    index = list(range(1, num_samples + 3))

    file_names.append(f"{base_name}_BLANK_1_V:1")
    file_names.append(f"{base_name}_BLANK_2_V:1")

    for loc in sample_locations:
        file_names.append(f"{base_name}_{loc}")
    
    st.session_state["sample_names"] = file_names

    inj_vols = [0.2] * (num_samples + 2)

    df = pd.DataFrame({
            "Index": index,
            "FILE_NAME": file_names,
            "INLET_FILE": [inlet_file] * (num_samples + 2),
            "SAMPLE_LOCATION": ["V:1","V:1"]+[f"{tray_no}:{loc[0]}.{loc[1:]}" for loc in sample_locations],
            "INJ_VOL": inj_vols,
            "INLET_SWITCH": [inlet_switch] * (num_samples + 2),
            "Is blank": [True, True] + [False] * num_samples,
            "Is marker": [False] * (num_samples + 2),
            "Is IS": [False] * (num_samples + 2),
            "Compound": ["" for _ in range(num_samples + 2)],
            "Compound Conc.": [0.0] * (num_samples + 2),
            "IS Conc.": [is_conc] * (num_samples + 2)
        })
    
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        key="editable_table"
    )
    
    # Convert DataFrame to CSV
    csv_data = edited_df.to_csv(index=False)
    st.download_button(
        label="Download Sample List CSV",
        data=csv_data,
        file_name="masslynx_sample_list.csv",
        mime="text/csv"
    )

# ----------- Tab 3: Single Sample Peak Deconvolution -----------
with tab3:
    # Step 1: Upload ZIP with raw folders
    st.header("🧪 Single Sample Peak Deconvolution")
    st.subheader("Step 1: Upload a ZIP containing two .raw folders (one sample, one blank)")
    zip_file = st.file_uploader("Upload ZIP", type=["zip"])

    # Step 2: Settings
    st.subheader("Step 2: Optional Settings")
    min_wavelength = st.number_input("Minimum Wavelength (nm)", value=220, step=1)
    min_time = st.number_input("Start Time (min)", value=0.25, step=0.01)
    max_time = st.number_input("End Time (min)", value=1.7, step=0.01)
    min_peak_height = st.number_input("Minimum Peak Relative Height", value=0.001, step=0.0001, format="%.4f", key="1")
    peak_model = st.selectbox(
        "Peak Model",
        ("FraserSuzuki", "BiGaussian", "BiGaussianTailing", "Bemg"), key="2"
    )
    r2_input = st.number_input("Peak Deconvolution Threshold", value=0.998, step=0.001, format="%.3f")

    if zip_file:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Detect .raw folders
            raw_folders = []
            for root, dirs, _ in os.walk(temp_dir):
                for d in dirs:
                    if d.lower().endswith(".raw"):
                        raw_folders.append(os.path.join(root, d))

            if len(raw_folders) < 2:
                st.error("❗ Please make sure your ZIP contains at least two .raw folders (sample and blank).")
            else:
                # Let user choose sample and blank folders
                folder_names = [os.path.basename(f) for f in raw_folders]
                sample_choice = st.selectbox("Select Sample .raw Folder", folder_names)
                blank_choice = st.selectbox("Select Blank .raw Folder", folder_names)

                sample_raw_folder = raw_folders[folder_names.index(sample_choice)]
                blank_raw_folder = raw_folders[folder_names.index(blank_choice)]

                if st.button("Run Deconvolution"):
                    with st.spinner("Processing..."):
                        try:
                            # Load chromatogram data
                            dataDirSample = rb.read(sample_raw_folder)
                            dataFileSample = dataDirSample.get_file("_FUNC001.DAT")
                            chromDataSample = Data2D(dataFileSample.xlabels, dataFileSample.ylabels, dataFileSample.data.T)

                            dataDirBlank = rb.read(blank_raw_folder)
                            dataFileBlank = dataDirBlank.get_file("_FUNC001.DAT")
                            chromDataBlank = Data2D(dataFileBlank.xlabels, dataFileBlank.ylabels, dataFileBlank.data.T)

                            # Create chromatogram
                            chromatogram = Chromatogram(chromDataSample, chromDataBlank)
                            chromatogram.correct_baseline()
                            chromatogram.extract_time(min_time, max_time, inplace=True)
                            chromatogram.extract_wavelength(min_wavelength, None, inplace=True)
                            chromatogram.find_peaks(min_rel_height=min_peak_height)

                            chromatogram.deconvolve_peaks(
                                model=peak_model, min_r2=r2_input, relaxe_concs=False, max_comps=5
                            )

                            st.success("✅ Deconvolution complete!")

                            # Plot with Plotly
                            ax = chromatogram.plot()
                            fig = ax.get_figure()
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"❌ Error during processing: {str(e)}")

# ----------- Tab 4: Batch Peak Deconvolution -----------
with tab4:
    # Misc functions
    def rename_duplicate_columns(df):
        seen = {}
        new_columns = []
        for col in df.columns:
            if col not in seen:
                seen[col] = 0
                new_columns.append(col)
            else:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
        df.columns = new_columns
        return df

    st.header("🧪🧪 Batch Peak Deconvolution and Quantification")

    # Step 1: Upload metadata and zip
    st.subheader("Step 1: Upload Files")
    metadata_file = st.file_uploader("Upload Metadata CSV", type=["csv"])
    raw_zip_file = st.file_uploader("Upload a ZIP containing .raw folders", type=["zip"])

    # Step 2: Settings
    st.subheader("Step 2: Optional Settings")
    min_elution_time = st.number_input("Minimum Elution Time", value=0.25, step=0.01, key="3")
    max_elution_time = st.number_input("Maximum Elution Time", value=1.7, step=0.1, key="4")
    min_wavelength = st.number_input("Minimum Wavelength", value=220, step=1, key="5")
    baseline_model = st.selectbox("Baseline Model", ["arpls", "asls", "flatfit"], key="6")
    min_peak_height_batch = st.number_input("Minimum Peak Relative Height", value=0.001, step=0.0001, format="%.4f", key="7")
    peak_model_batch = st.selectbox(
        "Peak Model",
        ("FraserSuzuki", "BiGaussian", "BiGaussianTailing", "Bemg"), key="8"
    )
    r2_input_batch = st.number_input("Peak Deconvolution Threshold", value=0.998, step=0.001, format="%.3f",key="29")

    # Step 3: Run processing
    if st.button("Run Mocca2 Processing") and metadata_file and raw_zip_file:
        with st.spinner("Processing LC data..."):
            metadata = pd.read_csv(metadata_file)

            # Extract zip to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "raw_data.zip")
                with open(zip_path, "wb") as f:
                    f.write(raw_zip_file.read())

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Recursively find all .raw folders (case-insensitive)
                extracted_dirs = {}
                for root, dirs, _ in os.walk(temp_dir):
                    for d in dirs:
                        if d.lower().endswith(".raw"):
                            extracted_dirs[d.lower()] = os.path.join(root, d)

                st.write("📂 Found .raw folders:", list(extracted_dirs.keys()))

                dataset = MoccaDataset()
                blankChrom = None

                for idx, row in metadata.iterrows():

                    sample_metadata = row
                    sample_name = str(sample_metadata["Sample Name"]).strip()
                    folder_key = (sample_name + ".raw").lower()
                    raw_path = extracted_dirs.get(folder_key)

                    if raw_path is None:
                        st.warning(f"❗ Could not find .raw folder for sample: {sample_name}")
                        continue

                    found_files = os.listdir(raw_path)
                    dataDir = rb.read(raw_path)

                    # Try _FUNC002.DAT first, fallback to _FUNC001.DAT
                    if "_FUNC002.DAT" in found_files:
                        dataFile = dataDir.get_file("_FUNC002.DAT")
                    elif "_FUNC001.DAT" in found_files:
                        dataFile = dataDir.get_file("_FUNC001.DAT")
                    else:
                        st.error(f"❌ Neither _FUNC002.DAT nor _FUNC001.DAT found for {sample_name}")
                        continue

                    chromData = Data2D(dataFile.xlabels, dataFile.ylabels, (dataFile.data).T)

                    if sample_metadata["Is blank"]:
                        blankChrom = chromData
                        continue

                    chrom = Chromatogram(chromData, blankChrom, name=sample_name)

                    if sample_metadata["Is IS"]:
                        dataset.add_chromatogram(
                            chrom,
                            reference_for_compound=sample_metadata["Compound"],
                            istd_reference=True,
                            compound_concentration=sample_metadata["IS Conc."],
                            istd_concentration=sample_metadata["IS Conc."]
                        )
                    elif sample_metadata["Is marker"]:
                        dataset.add_chromatogram(
                            chrom,
                            reference_for_compound=sample_metadata["Compound"],
                            compound_concentration=sample_metadata["Compound Conc."],
                            istd_concentration=sample_metadata["IS Conc."]
                        )
                    else:
                        dataset.add_chromatogram(
                            chrom,
                            istd_concentration=sample_metadata["IS Conc."]
                        )

                settings = ProcessingSettings(
                    baseline_model=baseline_model,
                    min_elution_time=min_elution_time,
                    max_elution_time=max_elution_time,
                    min_wavelength=min_wavelength,
                    explained_threshold=r2_input_batch,
                    min_rel_prominence = min_peak_height_batch,
                    peak_model = peak_model_batch
                )

                dataset.process_all(settings, verbose=True, cores=os.cpu_count())

                # Store the dataset in session_state
                st.session_state["dataset"] = dataset

    if "dataset" in st.session_state:
        
        results_type = st.selectbox("What results should be returned?", ("Peak areas", "Compound concentrations", "Compound concentrations relative to IS"))

        dataset = st.session_state["dataset"]

        if results_type == "Peak areas":
            results, _ = dataset.get_integrals()  # Extract the DataFrame from the tuple
            results = rename_duplicate_columns(results)
        elif results_type == "Compound concentrations":
            results, _ = dataset.get_concentrations()  # Extract the DataFrame from the tuple
            results = rename_duplicate_columns(results)
        elif results_type == "Compound concentrations relative to IS":
            results, _ = dataset.get_relative_concentrations()  # Extract the DataFrame from the tuple
            results = rename_duplicate_columns(results)
        else:
            st.warning("Please select a results type")
            st.stop()

        # Display results
        st.success("✅ Processing complete!")
        st.dataframe(results)

        # Export results to CSV
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results CSV", csv, "mocca2_results.csv", "text/csv")
        
        # Chromatogram visualization
        st.header("📈 Chromatogram Visualizations")
        chrom_names = []
        for chrom_id in dataset.chromatograms:
            chrom = dataset.chromatograms[chrom_id]
            chrom_names.append(chrom.name)
        selected_chroms = st.multiselect("Select chromatograms to display", chrom_names)

        for chrom_id in dataset.chromatograms:
            chromatogram = dataset.chromatograms[chrom_id]
            if chromatogram.name in selected_chroms:
                st.subheader(f"Chromatogram: {chromatogram.name}")
                ax = chromatogram.plot()
                fig = ax.get_figure()
                st.pyplot(fig)

# ----------- Tab 5: Data Visualisation -----------
with tab5:
    st.header("📊 MOCCA Peak Visualisation")

    # Upload files
    sample_list_file = st.file_uploader("Upload Sample List CSV", type="csv")
    results_file = st.file_uploader("Upload MOCCA Results CSV", type="csv")

    if sample_list_file and results_file:
        sample_list = pd.read_csv(sample_list_file)
        results = pd.read_csv(results_file)

        # Clean column names
        sample_list.columns = sample_list.columns.str.strip()
        results.columns = results.columns.str.strip()

        # Parse SAMPLE_LOCATION: X:A.Y → row letter (A–H), column number (1–12)
        def parse_sample_location(loc):
            match = re.match(r"[^:]*:([A-Z])\.(\d+)", str(loc))
            if match:
                return pd.Series([match.group(1), int(match.group(2))])
            else:
                return pd.Series([None, None])

        sample_list[["Row", "Column"]] = sample_list["SAMPLE_LOCATION"].apply(parse_sample_location)

        # Filter out blanks, IS, and markers
        mask = ~(sample_list["Is blank"] | sample_list["Is IS"] | sample_list["Is marker"])
        filtered_samples = sample_list[mask].copy()

        # Merge with MOCCA results
        merged = filtered_samples.merge(results, left_on="FILE_NAME", right_on="Chromatogram", how="inner")

        # Get peak columns from results
        data_columns = results.columns[2:]  # Assuming ID and Chromatogram are first two columns

        selected_peak = st.selectbox("Select peak to visualize:", data_columns)

        if selected_peak:
            # Prepare for plate-style grid heatmap
            row_labels = sorted(merged["Row"].dropna().unique())
            row_order = {row: i for i, row in enumerate(row_labels)}
            merged["Row_idx"] = merged["Row"].map(row_order)
            merged["Col_idx"] = merged["Column"]

            # Pivot to create heatmap data
            heatmap_data = pd.pivot_table(
                merged,
                values=selected_peak,
                index="Row_idx",
                columns="Col_idx",
                aggfunc="mean"
            )

            # Check if there's any data to plot
            if heatmap_data.empty or heatmap_data.isna().all().all():
                st.warning(f"No data available to plot for peak: {selected_peak}")
            else:
                # Replace index with row letters (A, B, etc.)
                heatmap_data.index = [row_labels[i] for i in heatmap_data.index]
                heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)

                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(
                    heatmap_data,
                    cmap="viridis",
                    annot=True,
                    fmt=".1f",
                    linewidths=0.5,
                    cbar_kws={"label": "Intensity"},
                    ax=ax
                )
                ax.set_title(f"Plate Heatmap for {selected_peak}")
                ax.set_xlabel("Column")
                ax.set_ylabel("Row")

                st.pyplot(fig)
