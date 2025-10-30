import streamlit as st
import pandas as pd
from mocca2 import MoccaDataset
import matplotlib.pyplot as plt
import re
import pickle
from datetime import datetime
import seaborn as sns
import numpy as np

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

def filter_positions(bool_df):
    rows = bool_df.index
    cols = bool_df.columns
    active_positions = [
        f"{row}{col}"
        for row in rows
        for col in cols
        if bool_df.loc[row, col]  # keep only True wells
    ]
    return active_positions

int_to_alphanum_48 = {}
alphanum_to_int_48 = {}
int_to_alphanum_96 = {}
alphanum_to_int_96 = {}

well_positions = generate_well_positions("48",48)
for i in range(0,48):
    int_to_alphanum_48[i+1] = well_positions[i]
    alphanum_to_int_48[well_positions[i]] = i+1

well_positions = generate_well_positions("96",96)
for i in range(0,96):
    int_to_alphanum_96[i+1] = well_positions[i]
    alphanum_to_int_96[well_positions[i]] = i+1

@st.cache_data
def clean_peak_areas(results):
    results.drop('Chromatogram ID', axis=1, inplace=True)
    results = rename_duplicate_columns(results)
    results.fillna(0.0, inplace=True)
    return results

@st.cache_data
def clean_peak_concs(results):
    results.drop('Chromatogram ID', axis=1, inplace=True)
    results = rename_duplicate_columns(results)
    results.fillna(0.0, inplace=True)
    return results

def custom_autopct(pct):
    return f'{pct:.1f}%' if pct > 0 else ''

# ---------- Streamlit App ----------
st.title("HTS Sample Submission and Visualisation Hub")

tab1, tab2 = st.tabs([
    "Sample List Generator",
    "Data Visualisation"
    ])

# ----------- Tab 1: MassLynx Sample List Generator -----------
with tab1:
    st.header("MassLynx Sample List Generator")
    
    tray_size = st.selectbox("Tray size", options=["48","96"], key="Sample submission tray szie")
    base_name = st.text_input("Base Sample Name", value="ELNXXXX-XXX_IPCX")
    lc_method = st.selectbox("LC Method", options=["Acidic", "Neutral"])
    tray_no = st.number_input("Tray No.", 1, 10)

    st.write(f"Using the checkboxes below, indicate which wells in the {tray_size}-well plate are in use.")

    if tray_size == "48":
        sample_array = np.ones((6,8), dtype=bool)
        sample_array_df = pd.DataFrame(sample_array, index=[f"{char}" for char in "ABCDEF"])
        sample_array_df.columns = [f"{_}" for _ in range(1,9)]
        
    elif tray_size == "96":
        sample_array = np.ones((8,12), dtype=bool)
        sample_array_df = pd.DataFrame(sample_array, index=[f"{char}" for char in "ABCDEFGH"])
        sample_array_df.columns = [f"{_}" for _ in range(1,13)]
    else:
        st.error("The tray size should be 48 or 96 for this feature to be used.")
    
    sample_wells = st.data_editor(
        sample_array_df,
        use_container_width=True,
        num_rows="fixed",
        key="editable_sample_well_table"
    )

    markers = st.toggle("Include markers (treat an internal/external standard as a marker)")

    if markers:
        include_qlc = st.toggle("Include quantitation")
        num_markers = int(st.number_input("Number of markers", min_value=1, max_value=None, value="min"))
        marker_df = pd.DataFrame()
        marker_df["Compound Name"] = [""] * num_markers
        marker_df.set_index(pd.Index([n+1 for n in range(0,num_markers)]), inplace=True)
        marker_table = st.data_editor(
            marker_df,
            use_container_width=True,
            num_rows="fixed",
            key="editable_marker_table"
        )

    # Assign method-specific values
    if lc_method == "Acidic":
        inlet_file = "C3S1_50_1p1"
        inlet_switch = "C3S1_50_switch_1p1"
    else:
        inlet_file = "C3S2_50_1p1"
        inlet_switch = "C3S2_50_switch_1p1"

    # Generate sample data
    file_names = []
    active_positions = filter_positions(sample_wells)
    num_unknowns = len(active_positions)
    num_samples = num_unknowns + 2 # include two blanks

    file_names.append(f"{base_name}_BLANK_1")
    file_names.append(f"{base_name}_BLANK_2")
    sample_locations = ["V:1","V:1"]
    is_marker_list = [False, False]
    compound_name_list = [""] * 2
    compound_conc_list = [0.0] * 2

    if markers:
        for idx, row in marker_table.iterrows():
            if include_qlc:
                concs = [0.1,0.2,0.5]
                for i in range(0,len(concs)):
                    num_samples += 1
                    file_names.append(f"{base_name}_{row["Compound Name"]}_qLC{i+1}")
                    compound_name_list.append(row["Compound Name"])
                    compound_conc_list.append(concs[i])
                    sample_locations.append("set appropriately")
                    is_marker_list.append(True)
            else:
                num_samples += 1
                file_names.append(f"{base_name}_{row["Compound Name"]}")
                compound_name_list.append(row["Compound Name"])
                compound_conc_list.append(0.0)
                sample_locations.append("set appropriately")
                is_marker_list.append(True)

    if tray_size == "48":
        for position in active_positions:
            sample_locations.append(f"{tray_no}:{alphanum_to_int_48[position]}")
    elif tray_size == "96":
        for position in active_positions:
            sample_locations.append(f"{tray_no}.{position[0]}:{position[1:]}")

    for loc in active_positions:
        file_names.append(f"{base_name}_{loc}")

    df = pd.DataFrame({
            "Index": [_+1 for _ in range(0, num_samples)],
            "FILE_NAME": file_names,
            "INLET_FILE": [inlet_file] * num_samples,
            "SAMPLE_LOCATION": sample_locations,
            "INJ_VOL": [0.2] * num_samples,
            "INLET_SWITCH": [inlet_switch] * num_samples,
            "Is blank": [True, True] + [False] * (num_samples - 2),
            "Is marker": is_marker_list + [False] * num_unknowns,
            "Compound": compound_name_list + [""] * num_unknowns,
            "Compound Conc.": compound_conc_list + [0.0] * num_unknowns
        })

    st.write("""The table below can be further edited as required. For markers, please set the location of the vials. 
             Any extra samples can be added at the bottom of the table - clicking the bottom grey row will create a new row.""")

    edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="editable_table"
        )

    # Convert DataFrame to CSV
    edited_df = edited_df.applymap(lambda x: re.sub(r'[\r\n]+', ' ', str(x)).strip() if isinstance(x, str) else x)
    csv_data = edited_df.to_csv(index=False, lineterminator='\r\n').encode('utf-8')
    st.download_button(
        label="Download Sample List CSV",
        data=csv_data,
        file_name=f"{base_name}_sample_list.csv",
        mime="text/csv"
    )

# ----------- Tab 2: Data Visualisation -----------
with tab2:
    st.header("Data Visualisation")

    data_source = st.radio("Select data source", ["Use peak area/concentration .pkl file", "Use full MOCCA dataset .pkl file"], index=0)

    if data_source == "Use peak area/concentration .pkl file":
        st.write("Upload a peak area/concetration dataset")
        peak_dataset = st.file_uploader("Upload peak area/concentration results .pkl file", type="pkl", key="peak info only file upload")
        if peak_dataset:
            peak_dataset = pickle.load(peak_dataset)
            st.session_state["peak area data"], _ = peak_dataset["peak areas"]
            st.session_state["peak conc data"], _ = peak_dataset["peak concentrations"]

    elif data_source == "Use full MOCCA dataset .pkl file":
        st.warning("Using a full MOCCA dataset may cause the Streamlit app to run very slowly. Use the peak area/concentration .pkl file where possible.")
        st.write("Upload a full MOCCA dataset")
        uploaded_mocca_dataset = st.file_uploader("Upload MOCCA results .pkl file", type="pkl", key="full mocca dataset file upload")
        if uploaded_mocca_dataset:
            st.write("Using uplaoded dataset")
            uploaded_mocca_dataset = MoccaDataset.from_dict(uploaded_mocca_dataset)
            st.session_state["peak area data"], _ = uploaded_mocca_dataset.get_intergrals()
            st.session_state["peak conc data"], _ = uploaded_mocca_dataset.get_concentrations()

    if "peak area data" in st.session_state and "peak conc data" in st.session_state:
        
        st.divider()

        st.header("Results Table")

        results_type = st.selectbox("What results should be returned?", ("Peak areas", "LCAP", "Compound concentrations"), index=1)
        
        if results_type == "Peak areas":
            peak_area_results = clean_peak_areas(st.session_state["peak area data"])
            st.dataframe(peak_area_results)
            csv = peak_area_results.to_csv(index=False).encode("utf-8")
        elif results_type == "Compound concentrations":
            concentration_results = clean_peak_concs(st.session_state["peak conc data"])
            st.dataframe(concentration_results)
            csv = concentration_results.to_csv(index=False).encode("utf-8")
        elif results_type == "LCAP":
            lcap_results = clean_peak_areas(st.session_state["peak area data"])
            # Identify retention time columns
            numeric_cols = lcap_results.select_dtypes(include='number').columns.tolist() # find numeric columns and convert to list

            metadata = lcap_results.drop(columns=numeric_cols) # splits into non-peak area info
            peak_area_data = lcap_results[numeric_cols] # splits into just numeric values

            # Creates a single-row DataFrame of checkboxes (True by default)
            col_selector_df = pd.DataFrame([True] * len(numeric_cols), index=numeric_cols).T 
            col_selector_df.index = ["Include"]

            st.write("Select Retention Times to Include in LCAP:")
            edited_selector = st.data_editor(
                col_selector_df,
                use_container_width=True,
                column_config={col: st.column_config.CheckboxColumn(required=True) for col in numeric_cols}
            )

            # Get selected columns from checkbox row
            selected_cols = [col for col in numeric_cols if edited_selector.iloc[0][col]]

            if not selected_cols:
                st.warning("Please select at least one retention time to calculate LCAP.")

            # Calculates LCAP from selected peaks
            selected_data = peak_area_data[selected_cols]
            total_areas = selected_data.sum(axis=1)
            lcap_data = selected_data.div(total_areas, axis=0) * 100

            # combines values with removed metadata
            final_df = pd.concat([metadata.reset_index(drop=True), lcap_data.reset_index(drop=True)], axis=1)
            lcap_results = final_df.round(1)
            st.dataframe(lcap_results)
            csv = lcap_results.to_csv(index=False).encode("utf-8")
        else:
            st.stop()

        if csv:
            st.download_button("Download table as csv", csv, f"{datetime.now()}_results_table.csv", "text/csv")

        st.divider()

        if st.toggle("Generate heatmap"):
            st.header("Heatmap")
            heatmap_type = st.selectbox("Which result set should be plotted?", ("LCAP", "Concentration", "Conversion"))

            heatmap_plate_size = st.selectbox("Plate size", options=["48","96"], key="heatmap_plate_size")
            if heatmap_plate_size == "48":
                n_heatmap_rows, n_heatmap_cols = 6, 8
                heatmap_row_labels = list("ABCDEF")
                heatmap_col_labels = list(range(1, 9))
            else:
                n_heatmap_rows, n_heatmap_cols = 8, 12
                heatmap_row_labels = list("ABCDEFGH")
                heatmap_col_labels = list(range(1, 13))

            if heatmap_type == "LCAP":
                try:
                    heatmap_results = lcap_results
                except:
                    st.warning("Please select the LCAP reuslts type from the drop down above to enable heatmap generation.")
            elif heatmap_type == "Concentration":
                st.error("Not implemented yet")
            elif heatmap_type == "Conversion":
                st.error("Not implemented yet")

            # Extract numeric columns only, skipping first 3
            numeric_cols = heatmap_results.columns[3:]

            selected_peak = st.selectbox("Select column to plot", numeric_cols)

            # Initialize empty grid
            heatmap_array = np.full((n_heatmap_rows, n_heatmap_cols), np.nan)

            # Process each row
            for _, row in heatmap_results.iterrows():
                sample_name = str(row["Chromatogram"])
                match = re.search(r'(\D+)(\d+)', sample_name.split('_')[-1])
                if match:
                    row_letter, col_number = match.groups()
                    row_idx = ord(row_letter.upper()) - ord('A')
                    col_idx = int(col_number) - 1
                    if 0 <= row_idx < n_heatmap_rows and 0 <= col_idx < n_heatmap_cols:
                        heatmap_array[row_idx, col_idx] = row[selected_peak]

            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(n_heatmap_cols, n_heatmap_rows // 2 + 2))
            custom_cmap = sns.light_palette("seagreen", as_cmap=True)
            sns.heatmap(
                heatmap_array,
                annot=True,
                fmt=".2f",
                cmap=custom_cmap,
                cbar_kws={"label": f"{heatmap_type}"},
                linewidths=0.5,
                linecolor="white",
                ax=ax,
                vmin=0,
                vmax=100
            )
            ax.set_title(f"Heatmap of {selected_peak} {heatmap_type}")
            ax.tick_params(axis=u'both', which=u'both', length=0)
            ax.set_xticks([n+0.5 for n in range(0,heatmap_array.shape[1])], heatmap_col_labels)
            ax.set_yticks([n+0.5 for n in range(0,heatmap_array.shape[0])], heatmap_row_labels, rotation="horizontal")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")

            st.pyplot(fig)
            
            st.divider()

        if st.toggle("Generate array of pie charts"):
            st.header("Pie Chart Array")

            try:
                pie_chart_results = lcap_results.drop("Chromatogram", axis=1)
                pie_chart_sample_names = lcap_results["Chromatogram"].tolist()
            except KeyError:
                st.warning("Please select the LCAP results type from the dropdown above to enable pie chart generation.")
                st.stop()

            st.write("Select columns to include explicitly in the pie chart. All other columns will be summed together as 'Total Others'.")
            default_colours = ['#118ab2','#06d6a0','#ffd166','#dabfff','#f48c06','#ff8fa3']

            num_components = st.number_input("Number of compounds", min_value=1, value=2)
            components, colours = [], []

            for x in range(num_components):
                col1, col2 = st.columns([2,1])
                with col1:
                    component_name = st.selectbox(f"Compound {x+1}", pie_chart_results.columns, key=f"comp_{x}")
                    components.append(component_name)
                with col2:
                    default_colour = default_colours[x % len(default_colours)]
                    colour = st.color_picker("Choose a colour", default_colour, key=f"col_{x}")
                    colours.append(colour)

            all_columns = set(pie_chart_results.columns)
            unselected_columns = list(all_columns - set(components))

            if unselected_columns:
                col1, col2 = st.columns([3,1])
                with col1:
                    others = st.text_input('Rename Total Others?', 'Total Others')
                with col2:
                    oth_colour = st.color_picker('Pick a colour', '#ff8fa3')
                pie_chart_results[others] = pie_chart_results[unselected_columns].sum(axis=1)
                components.append(others)
                colours.append(oth_colour)

            pie_chart_labels = st.toggle('Include value labels')
            pie_chart_label_size = st.number_input('Input label font size', min_value=1, max_value=100, value=15) if pie_chart_labels else None

            plate_size = st.selectbox("Plate size", options=["48","96"], key="pie_chart_plate_size")
            if plate_size == "48":
                n_rows, n_cols = 6, 8
                default_row_labels = list("ABCDEF") + ["not used"] * 2
                default_col_labels = [str(n) for n in range(1,9)]
            else:
                n_rows, n_cols = 8, 12
                default_row_labels = list("ABCDEFGH") + ["not used"] * 4
                default_col_labels = [str(n) for n in range(1,13)]

            # Editable labels
            row_col_labels_df = pd.DataFrame({"Row Labels": default_row_labels, "Col Labels": default_col_labels})
            if st.toggle("Specify plate row and column labels"):
                row_col_labels_table = st.data_editor(row_col_labels_df, use_container_width=True, num_rows="fixed", key="editable_labels")
            else:
                row_col_labels_table = row_col_labels_df

            # Customisation
            title = st.text_input("Input plot title", "")
            title_size = st.number_input('Font size', min_value=1, max_value=50, value=20)
            size_label = st.number_input('Row/Col label font size', min_value=1, max_value=100, value=15)
            dis = st.number_input('Column padding', min_value=-3, max_value=50, value=20)
            legend_size = st.number_input('Legend font size', min_value=1, max_value=100, value=15)
            pad = st.number_input('Legend padding', min_value=-3.00, max_value=5.00, value=0.00)

            # Prepare subplot grid
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.5*n_cols, 1.5*n_rows))
            axs = np.atleast_2d(axs)  # ensure 2D indexing

            if st.button("Create Pie Chart Array"):
                # Precompute well positions
                well_matches = [re.search(r'(\D+)(\d+)', sn.split('_')[-1]) for sn in pie_chart_sample_names]
                valid_positions = []
                for idx, row in enumerate(pie_chart_results[components].to_numpy()):
                    match = well_matches[idx]
                    if match:
                        row_letter, col_number = match.groups()
                        row_idx = ord(row_letter.upper()) - ord('A')
                        col_idx = int(col_number) - 1
                        valid_positions.append((row_idx,col_idx))
                        if 0 <= row_idx < n_rows and 0 <= col_idx < n_cols:
                            ax = axs[row_idx, col_idx]
                            if pie_chart_labels:
                                wedges, texts, autotexts = ax.pie(
                                    row,
                                    colors=colours,
                                    startangle=90,
                                    counterclock=False,
                                    wedgeprops={'linewidth': 0.5},
                                    autopct=custom_autopct,
                                )
                                for value, text in zip(row, autotexts):
                                    text.set_fontsize(pie_chart_label_size)
                                    if value < 25:
                                        x, y = text.get_position()
                                        text.set_position((x * 2.3, y * 2.3))
                                        text.set_ha("center")
                                        text.set_color("black")
                                        ax.annotate('', xy=(x, y), xytext=(x * 1.9, y * 1.9),
                                                    arrowprops=dict(arrowstyle='-', color='black', lw=0.5))
                            else:
                                ax.pie(row, colors=colours, startangle=90, counterclock=False, wedgeprops={'linewidth': 0.5})

                for i in range(0,axs.shape[0]):
                    for j in range(0,axs.shape[1]):
                        if (i,j) not in valid_positions:
                            ax = axs[i,j]
                            ax.set_facecolor('#f0f0f0')
                            ax.set(aspect='equal')
                            ax.axis('off')

                # Column labels
                for j, col_lab in enumerate(row_col_labels_table["Col Labels"].values[:n_cols]):
                    axs[0, j].set_title(str(col_lab), fontsize=size_label, pad=dis)

                # Row labels
                for i, row_lab in enumerate(row_col_labels_table["Row Labels"].values[:n_rows]):
                    axs[i, 0].annotate(str(row_lab), xy=(-0.4, 0.5), xycoords='axes fraction',
                                    ha='right', va='center', fontsize=size_label)

                plt.tight_layout()
                plt.subplots_adjust(left=0.15, top=0.92, bottom=0.08)
                plt.suptitle(title, fontsize=title_size)
                fig.legend(components, loc='lower center', bbox_to_anchor=(0.5, pad), ncol=3,
                        fontsize=legend_size, frameon=False)

                st.pyplot(fig)
