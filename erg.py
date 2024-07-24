import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import os
from glob import glob

# Define the Naka-Rushton function
def naka_rushton(x, V_max, k, n):
    try:
        return V_max * (x**n / (k**n + x**n))
    except OverflowError:
        return np.inf

# Helper function to process a single data file
def processing_a_b_wave(file_path):
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        marker_data = data.iloc[2:, 9:21]
        marker_data.columns = [
            'Group', 'Name', 'Cage #', 'Age', 'Date', 
            'Step', 'Channel', 'Repetition', 'Eye', 'Wave_Type', 
            'Amplitude_uV', 'Time_ms'
        ]
        marker_data = marker_data.drop(columns=['Name', 'Cage #'])
        marker_data['Age'] = marker_data['Age'].str.extract('(\d+)').astype(float)
        marker_data['Amplitude_uV'] = pd.to_numeric(marker_data['Amplitude_uV'], errors='coerce')
        marker_data = marker_data.dropna()

        stimulus_data = data.iloc[2:, 32:35]
        stimulus_data.columns = ['Step', "Description", 'Flash_Intensity']
        stimulus_data = stimulus_data.dropna()
        stimulus_data['Log_Flash_Intensity'] = stimulus_data['Flash_Intensity'].apply(
            lambda x: np.log10(float(x.split()[0]))
        )
        
        merged_data = pd.merge(marker_data, stimulus_data, on='Step', how='left')
        merged_data = merged_data[merged_data['Step'] != '7']
        merged_data.loc[merged_data['Wave_Type'] == 'a', 'Amplitude_uV'] *= -1
        
        return merged_data
    
    except Exception as e:
        st.error(f"An error occurred while processing {file_path}: {e}")
        return pd.DataFrame()

# Function to process uploaded files
def process_uploaded_files(uploaded_files):
    all_data = []
    
    for uploaded_file in uploaded_files:
        processed_data = processing_a_b_wave(uploaded_file)
        if not processed_data.empty:
            filename = os.path.basename(uploaded_file.name)
            mouse_id, experiment_setup = filename.split('.')[0].split('_')[1:3]
            processed_data['Mouse_ID'] = mouse_id
            processed_data['Experiment_Setup'] = experiment_setup
            all_data.append(processed_data)
        else:
            st.warning(f"Skipping file (no data or format issues): {uploaded_file.name}")
    
    if not all_data:
        st.error("No valid data found in any uploaded files.")
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

# Plotting function
def plot_wave_data_and_fit_naka_rushton(data, wave_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")
    setups = data['Experiment_Setup'].unique()
    palette = sns.color_palette("bright", len(setups))

    for setup, color in zip(setups, palette):
        setup_data = data[(data['Experiment_Setup'] == setup) & (data['Wave_Type'] == wave_type)]
        grouped = setup_data.groupby('Log_Flash_Intensity').agg(
            mean_amplitude=('Amplitude_uV', 'mean'),
            sem_amplitude=('Amplitude_uV', 'sem')
        ).reset_index()

        if grouped.empty:
            st.warning(f"No data available for {wave_type}-wave in setup: {setup}")
            continue

        ax.errorbar(grouped['Log_Flash_Intensity'], grouped['mean_amplitude'], yerr=grouped['sem_amplitude'],
                    fmt='o-', color=color, label=f'{setup}', capsize=5)

        try:
            params, _ = curve_fit(
                naka_rushton, grouped['Log_Flash_Intensity'], grouped['mean_amplitude'],
                p0=[np.max(grouped['mean_amplitude']), np.median(grouped['Log_Flash_Intensity']), 1.0],
                maxfev=10000
            )
            x_model = np.linspace(grouped['Log_Flash_Intensity'].min(), grouped['Log_Flash_Intensity'].max(), 400)
            y_model = naka_rushton(x_model, *params)
            ax.plot(x_model, y_model, '--', color=color, label=f'Fit: {setup}')
        except Exception as e:
            st.error(f"Error fitting curve for {setup}: {e}")

    ax.set_xlabel('Log Flash Intensity (log cd.s/m²)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title(f'{wave_type}-Wave Amplitude vs. Log Flash Intensity')
    ax.legend()
    st.pyplot(fig)

# Main app function
def app():
    st.title('Wave Data Processing and Visualization')

    uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type=['csv'])
    wave_type = st.selectbox('Select Wave Type', options=['a', 'B'])

    if st.button('Process and Plot') and uploaded_files:
        all_data = process_uploaded_files(uploaded_files)

        if not all_data.empty:
            plot_wave_data_and_fit_naka_rushton(all_data, wave_type)
        else:
            st.error("Failed to process any data.")

if __name__ == "__main__":
    app()

