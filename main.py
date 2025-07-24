# Import Modules
import requests
import sys
import os
import io
import folium
import subprocess
import leafmap.foliumap as leafmap
from io import BytesIO
import base64
from folium.plugins import MarkerCluster
from PyQt5.QtGui import QFont, QDoubleValidator
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import ( QMainWindow, QApplication, QWidget, QFileDialog, QDialog, 
            QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, 
            QGridLayout, QLabel, QListWidget, QComboBox, QTextBrowser,
            QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QColor
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QThread, pyqtSignal  # Import QThread and pyqtSignal
from PyQt5.QtGui import QFont
from PIL import Image, ImageFilter, ImageEnhance

import pandas as pd
import numpy as np
import seaborn as sns
from pylab import *
import plotly.express as px
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.figure import Figure
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
import contextily as ctx
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseEvent
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d, convolve, sobel, median_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from skimage import exposure
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm, ListedColormap, LightSource
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# ADTK Imports
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
from adtk.detector import OutlierDetector
from adtk.detector import VolatilityShiftAD
from scipy.stats import zscore
from scipy.signal import medfilt
from sklearn.neighbors import LocalOutlierFactor
import webbrowser  # Import the webbrowser module

class MagnavisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.file_paths = {}  # Store file paths with filename as key

    def initUI(self):
        self.setWindowTitle("MagNavis")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: lightblue;")

        self.data_files = []  # Store selected files
        self.current_data = None
        self.file_paths = {}  # Store full path with filename as key

        # Widgets

        #Import File Button
        self.import_file = QPushButton("Import File")
        self.import_file.setGeometry(50, 50, 100, 40)
        self.import_file.setStyleSheet('background-color: red; border: 1px outset black;')
        self.import_file.setFont(QFont('Arial', 12))
        self.import_file.clicked.connect(self.open_file)

        #File List Widget
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.display_file_data)
    
        # Dropdown Box
        self.label_box1 = QLabel("Data Insights")
        self.label_box1.setFont(QFont('Arial', 12))

        self.filter_box1 = QComboBox()
        self.filter_box1.setFont(QFont('Arial', 10))
        self.filter_box1.setStyleSheet('background-color: white;')
        #filter_box.addItem("Data Insights")
        self.filter_box1.addItems(["Select Insight", "Data Details","CDF Analysis", "Frequency Spectrum", 
                                   "Power Spectral Density","Time Series"  ])
        self.filter_box1.currentIndexChanged.connect(self.process_data)
      
        self.label_box2 = QLabel("Anomaly Detection")
        self.label_box2.setFont(QFont('Arial', 12))

        self.anomaly_detection_method_dropdown = QComboBox()
        self.anomaly_detection_method_dropdown.setFont(QFont('Arial', 10))
        self.anomaly_detection_method_dropdown.setStyleSheet('background-color: white;')
        # Dictionary of function descriptions and URLs
        self.anomaly_detection_method_descriptions = {
            "Select Method": ( ),

            "Threshold Detection": (
                "<p style='font-family: Arial; font-size: 12pt;'>"
                "Threshold-based anomaly detection identifies data points that exceed predefined upper or lower bounds.\n"
                "It compares each time series value with given thresholds.\n"
                "It is a simple yet effective method for detecting extreme values in a dataset.\n"
                "This method works well when normal values fall within a known range.\n"
                "However, it may fail to detect subtle anomalies that don’t cross the threshold.\n"
                "<a href='https://arundo-adtk.readthedocs-hosted.com/en/stable/examples.html'>Learn more</a>"
                "</p>"
            ),

            "Quantile Detection": (
                "<p style='font-family: Arial; font-size: 12pt;'>"
                "Quantile-based anomaly detection identifies anomalies by setting thresholds based on statistical quantiles.\n"
                "It compares each time series value with historical quantiles.\n"
                "It is useful when data distributions are skewed and fixed thresholds don’t work well.\n"
                "Lower quantiles capture small values, while upper quantiles capture extreme values.\n"
                "It provides a robust way to handle data with outliers.\n"
                "<a href='https://arundo-adtk.readthedocs-hosted.com/en/stable/examples.html'>Learn more</a>"
                "</p>"
            ),

            "Inter Quartile Range Detection": (
                "<p style='font-family: Arial; font-size: 12pt;'>"
                "IQR-based anomaly detection identifies outliers using the interquartile range (IQR) which is based on based on simple historical statistics.\n"
                "When a value is out of the range defined by [Q1−c×IQR, Q3+c×IQR] where IQR=Q3−Q1 is the difference between 25% and 75% quantiles.\n"
                "It considers values that lie significantly outside the middle 50% of the dataset as anomalies.\n"
                "This method is widely used in statistical data analysis.\n"
                "This detector is usually preferred to QuantileAD in the case where only a tiny portion or even none of training data is anomalous.\n"
                "IQR is particularly effective when dealing with non-normal data distributions.\n"
                "<a href='https://arundo-adtk.readthedocs-hosted.com/en/stable/examples.html'>Learn more</a>"
                "</p>"
            ),

            "ESD Detection": (
                "<p style='font-family: Arial; font-size: 12pt;'>"
                """"Generalized Extreme Studentized Deviate (ESD) test is used to detect one or more outliers in a dataset.
                It detects anomaly based on generalized extreme Studentized deviate (ESD) test.
                It iteratively removes the most extreme value and checks if it is significantly different from the rest.
                ESD is particularly useful for detecting a small number of anomalies in a large dataset.
                It assumes data follows a normal distribution, which may limit its effectiveness in some cases.
                It compares each time series value with its previous values. Internally, it is implemented as a pipenet with transformer DoubleRollingAggregate."""

                """In Generalized ESD (Extreme Studentized Deviate) Test, α (alpha) is the significance level, which controls how strictly anomalies are identified.
                What Does Alpha Do?\n\n
                1. It represents the probability of falsely identifying a normal data point as an anomaly (Type I Error).\n\n
                2. Lower alpha (e.g., 0.01 or 0.05) → More conservative detection (fewer anomalies, but higher confidence).\n\n
                3. Higher alpha (e.g., 0.1 or 0.3) → More lenient detection (more anomalies, but higher false positives).\n\n"""
                "<a href='https://arundo-adtk.readthedocs-hosted.com/en/stable/examples.html'>Learn more</a>"
                "</p>"
            ),

            "PersistAD": (
                "<p style='font-family: Arial; font-size: 12pt;'>"
                "Persistence-based anomaly detection identifies anomalies by analyzing long-term trends in the data.\n"
                "It compares each time series value with its previous values. Internally, it is implemented as a pipenet with transformer DoubleRollingAggregate.\n"
                "It detects unusual patterns based on how consistently values deviate from expected behavior.\n"
                "It is particularly useful for identifying gradual anomalies rather than sudden spikes.\n"
                "PersistAD is effective in detecting drift in sensor data or other time-series datasets.\n"
                "<a href='https://arundo-adtk.readthedocs-hosted.com/en/stable/examples.html'>Learn more</a>"
                "</p>"
            ),

            "LevelShiftAD": (
                "<p style='font-family: Arial; font-size: 12pt;'>"
                "Level shift anomaly detection identifies sudden shifts in the baseline level of a dataset.\n"
                "It is useful for detecting structural changes in time-series data, such as an abrupt increase or decrease.\n"
                "This method helps in identifying changes caused by external factors, such as equipment failure or policy changes.\n"
                "The algorithm works by monitoring windows of data and checking for significant shifts.\n"
                "<a href='https://arundo-adtk.readthedocs-hosted.com/en/stable/examples.html'>Learn more</a>"
                "</p>"
            )

        }
        self.anomaly_detection_method_dropdown.addItems(self.anomaly_detection_method_descriptions.keys())
        self.anomaly_detection_method_dropdown.currentIndexChanged.connect(self.update_inputs)

        # Description Box (QTextBrowser for clickable links)
        self.description_box = QTextBrowser()
        self.description_box.setStyleSheet("background-color: lightyellow; border: 1px solid black; padding: 5px;")
        self.description_box.setOpenExternalLinks(True)

        # Labels and LineEdits for parameter selection for Anomaly Detection
        self.param1_label = QLabel("Parameter 1:")
        self.param1_input = QLineEdit()
        self.param1_input.setValidator(QDoubleValidator(0.0, 100000.0, 2))  # Float values
        self.param1_input.setPlaceholderText("Enter value...")
        
        self.param2_label = QLabel("Parameter 2:")
        self.param2_input = QLineEdit()
        self.param2_input.setValidator(QDoubleValidator(0.0, 100000.0, 2))
        self.param2_input.setPlaceholderText("Enter value...")
        
        # Submit Button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.anomaly_detection)

        # Dropdown Box
        self.tool_label_box = QLabel("Data Processing Tools")
        self.tool_label_box.setFont(QFont('Arial', 12))

        self.tool_box = QComboBox()
        self.tool_box.setFont(QFont('Arial', 10))
        self.tool_box.setStyleSheet('background-color: white;')
        #filter_box.addItem("Data Insights")
        self.tool_box.addItems(["Select Method", "Low Pass Filter", "Time Lag Correction", 
                                "Residual Mag Anomaly", "Analytic Signal Generation"])
        self.tool_box.currentIndexChanged.connect(self.processing_tool)

        self.label_box3 = QLabel("Visualisation")
        self.label_box3.setFont(QFont('Arial', 12))

        self.filter_box3 = QComboBox()
        self.filter_box3.setFont(QFont('Arial', 10))
        self.filter_box3.setStyleSheet('background-color: white;')
        self.filter_box3.addItem("Select Visualisation")
        self.filter_box3.addItem("Anomaly Data Points Plot")
        self.filter_box3.addItem("Residual Anomaly Plot")
        self.filter_box3.addItem("Analytic Signal (AS) Plot")
        self.filter_box3.addItem("Residual Anomaly & AS")
        self.filter_box3.addItem("AS 3D Plot")
        self.filter_box3.addItem("AS on Map")
        self.filter_box3.currentIndexChanged.connect(self.update_inputs_vis)

        # Labels and LineEdits for parameter selection for Visualisation
        self.param3_label = QLabel("Sigma (Gaussian Filter):")
        self.param3_input = QLineEdit()
        self.param3_input.setValidator(QDoubleValidator(0.1, 10, 2))  # Float values
        self.param3_input.setPlaceholderText("Enter value...")
        
        self.param4_label = QLabel("Size (Median Filter):")
        self.param4_input = QLineEdit()
        self.param4_input.setValidator(QDoubleValidator(1, 50, 2))
        self.param4_input.setPlaceholderText("Enter value...")

        self.param5_label = QLabel("Colour Normalisation:")
        self.param5_input = QLineEdit()
        self.param5_input.setValidator(QDoubleValidator(1.0, 10.0, 2))
        self.param5_input.setPlaceholderText("Enter value...")
        
        # Submit Button
        self.submit_button_vis = QPushButton("Submit")
        self.submit_button_vis.clicked.connect(self.vis_method)

        # Select Columns Button
        self.btn_select_columns = QPushButton("Select Columns")
        self.btn_select_columns.setGeometry(50, 50, 100, 40)
        self.btn_select_columns.setStyleSheet('background-color: yellow; border: 1px outset black;')
        self.btn_select_columns.setFont(QFont('Arial', 12))
        self.btn_select_columns.clicked.connect(self.open_column_selection)

        #Import File Button
        
        #self.btn_heatmap = QPushButton("Heat Map")
        #self.btn_heatmap.setGeometry(50, 50, 100, 40)
        #self.btn_heatmap.setStyleSheet('background-color: orange; border: 1px outset black;')
        #self.btn_heatmap.setFont(QFont('Arial', 12))
#
        #self.btn_visonmap = QPushButton("Visualisation on Map")
        #self.btn_visonmap.setGeometry(50, 50, 100, 40)
        #self.btn_visonmap.setStyleSheet('background-color: lightgreen; border: 1px outset black;')
        #self.btn_visonmap.setFont(QFont('Arial', 12))

        # App Design
        master_layout = QHBoxLayout()

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        col1.addWidget(self.import_file)
        col1.addWidget(self.file_list)
        col1.addWidget(self.btn_select_columns)
        col1.addWidget(self.label_box1, alignment=Qt.AlignCenter)
        col1.addWidget(self.filter_box1)
        col1.addWidget(self.label_box2, alignment=Qt.AlignCenter)
        col1.addWidget(self.anomaly_detection_method_dropdown)
        col1.addWidget(self.param1_label)
        col1.addWidget(self.param1_input)
        col1.addWidget(self.param2_label)
        col1.addWidget(self.param2_input)
        col1.addWidget(self.submit_button)
        col1.addWidget(self.tool_label_box, alignment=Qt.AlignCenter)
        col1.addWidget(self.tool_box)
        col1.addWidget(self.label_box3, alignment=Qt.AlignCenter)
        col1.addWidget(self.filter_box3)
        col1.addWidget(self.param3_label)
        col1.addWidget(self.param3_input)
        col1.addWidget(self.param4_label)
        col1.addWidget(self.param4_input)
        col1.addWidget(self.param5_label)
        col1.addWidget(self.param5_input)
        col1.addWidget(self.submit_button_vis)
        #col1.addWidget(self.btn_heatmap)
        #col1.addWidget(self.btn_visonmap)

        # Hide second parameter initially
        self.param1_label.hide()
        self.param1_input.hide()
        self.param2_label.hide()
        self.param2_input.hide()
        self.submit_button.hide()
        self.description_box.hide()

        self.param3_label.hide()
        self.param3_input.hide()
        self.param4_label.hide()
        self.param4_input.hide()
        self.param5_label.hide()
        self.param5_input.hide()
        self.submit_button_vis.hide()

        # ✅ **Create Logo QLabel**
        self.logo_label = QLabel(self)
        pixmap = QPixmap(r"C:\Users\kundu\Desktop\GUI Making (2)\Magnavis\redlogo.jpg")  
        self.logo_label.setPixmap(pixmap)
        #self.logo_label.setScaledContents(True)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("background-color: white;")

        #Table Widget To Display Data
        self.table = QTableWidget()
        self.rendering_area = QLabel("Rendering Area")
        self.canvas = FigureCanvas(plt.Figure())
        self.canvas.hide()  
        self.table.hide()
        self.rendering_area.hide()

        col2.addWidget(self.table, 20)
        col2.addWidget(self.logo_label)  # Logo shown initially
        col2.addWidget(self.canvas, 80)
        col2.addWidget(self.description_box)
        col2.addWidget(self.rendering_area)

        master_layout.addLayout(col1, 15)
        master_layout.addLayout(col2, 85)

        self.setLayout(master_layout)
        self.map_view = QWebEngineView()
        #self.btn_heatmap.clicked.connect(self.generate_heatmap)
        #self.btn_visonmap.clicked.connect(self.plot_map)

# Open file dialog to select a CSV file

    def open_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xls *.xlsx)", options=options)
        
        if file_path:
            file_name = os.path.basename(file_path)  # Extract only file name
            #file_name = file_path.split("/")[-1]  # Extract file name
            if file_name not in self.file_paths:
                self.file_paths[file_name] = file_path  # Store full path
                self.file_list.addItem(file_name)  # Add only file name to list    

    def display_file_data(self, item):

        self.canvas.show()  
        self.table.show()
        self.rendering_area.show()
        self.logo_label.hide()

        file_name = item.text()
        file_path = self.file_paths.get(file_name)
        
        if file_path:
            if file_path.endswith(".csv"):
                #data = pd.read_csv(file_path)
                try:
                    # Step 1: Read the first (header) line to count actual columns
                    with open(file_path, 'r') as f:
                        header_line = f.readline().strip()

                    # Step 2: Count number of non-empty column headers
                    num_cols_to_read = len(header_line.split(','))

                    # Step 3: Read only that many columns using pandas
                    data = pd.read_csv(
                        file_path,
                        usecols=range(num_cols_to_read),
                        header=0,  # Use first row as header
                        sep=','
                    )

                    print(data.head())

                except FileNotFoundError:
                    print(f"Error: File not found at {file_path}")
                except Exception as e:
                    print(f"An error occurred: {e}")
            else:
                data = pd.read_excel(file_path)
            
            print(f"Loaded data type: {type(data)}")  # Debugging output
            self.data = data
            self.update_table(data)

    def update_table(self, data):
        self.table.clear()
        self.table.setRowCount(min(25, len(data)))  # Show only 25 rows
        self.table.setColumnCount(len(data.columns))
        self.table.setHorizontalHeaderLabels(data.columns)

        for row in range(min(25, len(data))):
            for col in range(len(data.columns)):
                self.table.setItem(row, col, QTableWidgetItem(str(data.iat[row, col])))

    def get_selected_file_path(self):
        """Retrieve the currently selected file path from the file list."""
        selected_item = self.file_list.currentItem()  # Get selected item

        if selected_item:
            file_name = selected_item.text()
            return self.file_paths.get(file_name, None)  # Get full path from dictionary
        return None

    def open_column_selection(self):
        file_path = self.get_selected_file_path()  # Get selected file

        if file_path:  # Ensure file exists
            try:
                # Step 1: Read the first (header) line to count actual columns
                with open(file_path, 'r') as f:
                    header_line = f.readline().strip()

                    # Step 2: Count number of non-empty column headers
                num_cols_to_read = len(header_line.split(','))

                    # Step 3: Read only that many columns using pandas
                data = pd.read_csv(
                    file_path,
                    usecols=range(num_cols_to_read),
                    header=0,  # Use first row as header
                    sep=','
                )
                print("Data Loaded Successfully:\n", data.head())  # Debugging
                print(data.head())

            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
            except Exception as e:
                print(f"An error occurred: {e}")    

            #try:
            #    data = pd.read_csv(file_path)  # Read CSV
            #    print("Data Loaded Successfully:\n", data.head())  # Debugging
            #except Exception as e:
            #    print("Error loading file:", e)
            #    return
        else:
            print("Error: File not found. Check file path.")

        columns = self.data.columns  # Get column names
        self.column_dialog = QDialog(self)
        self.column_dialog.setWindowTitle("Select Columns")
        self.column_dialog.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        self.dropdowns = {}
        labels = ["Latitude", "Longitude", "Magnetic_Values", "Date", "Time", "Altitude"]

        for label in labels:
            row_layout = QHBoxLayout()
            lbl = QLabel(label)
            combo = QComboBox()
            combo.addItem("NIL")  # Allow skipping selection
            combo.addItems(columns)  # Populate dropdown with column names
            self.dropdowns[label] = combo
            row_layout.addWidget(lbl)
            row_layout.addWidget(combo)
            layout.addLayout(row_layout)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.clicked.connect(self.save_selected_columns)
        layout.addWidget(self.submit_btn)

        self.column_dialog.setLayout(layout)
        self.column_dialog.exec_()
       
    def save_selected_columns(self):
        self.selected_columns = {
            key: dropdown.currentText() if dropdown.currentText() != "NIL" else None
            for key, dropdown in self.dropdowns.items()
        }
        #print("Selected Columns:", self.selected_columns)  # Debugging
        self.column_dialog.accept()  # Close dialog

    def extract_selected_data(self):
        
        """Extracts only the selected columns from the loaded dataset."""
    
        if self.data is None:
            print("Error: No data loaded. Ensure a file is selected and loaded before extracting columns.")
            return None  

        # Standardize column names by stripping spaces
        self.data.columns = self.data.columns.str.strip()

        # Create a mapping of user-selected column names to standardized names
        rename_mapping = {
            self.selected_columns.get("Longitude", "").strip(): "Longitude",
            self.selected_columns.get("Latitude", "").strip(): "Latitude",
            self.selected_columns.get("Magnetic_Values", "").strip(): "Magnetic_Values",
            self.selected_columns.get("Altitude", "").strip(): "Altitude",
            self.selected_columns.get("Time", "").strip(): "Time",
            self.selected_columns.get("Date", "").strip(): "Date"
        }

        # Remove any empty keys from mapping
        rename_mapping = {k: v for k, v in rename_mapping.items() if k and k in self.data.columns}

        if not rename_mapping:
            print("Error: No valid columns selected.")
            return None

        # Rename the selected columns in the DataFrame
        self.filtered_data = self.data[list(rename_mapping.keys())].rename(columns=rename_mapping)#.dropna()

        #print("Filtered & Renamed Data Columns:", self.filtered_data.columns.tolist())  # Debugging output

        #print(self.filtered_data.head())

        return self.filtered_data
    
    def process_data(self):

        self.canvas.show()  
        self.table.show()
        self.rendering_area.show()
        self.logo_label.hide()

        self.filtered_data = self.extract_selected_data()

        if self.filtered_data is not None:
            if len(self.filtered_data.columns) >= 1:
                # Extract the selected analysis type
                self.description_box.hide()
                analysis_type1 = self.filter_box1.currentText()

                if analysis_type1 == "Data Details":
                    # Clear the previous plot
                    self.canvas.figure.clear()

                    # Extract data
                    mag_values = self.filtered_data['Magnetic_Values'].dropna().sort_values()
                    total_rows = len(mag_values)
                    total_columns = len(self.data.columns)
                    #print(f"Total Number of Columns = {total_columns}") #successfully printing
                    #print(f"Total Number of Rows (mag_values) = {total_rows}") #successfully printing
                    # Compute top and bottom 5% counts
                    top_3_percent_index = int(0.99 * total_rows)  # 95% index for top 5%
                    bottom_3_percent_index = int(0.01 * total_rows)  # 5% index for bottom 5%

                    top_3_percent_values = mag_values.iloc[top_3_percent_index:]
                    bottom_3_percent_values = mag_values.iloc[:bottom_3_percent_index]

                    total_top_3 = top_3_percent_values.min()
                    total_bottom_3 = bottom_3_percent_values.max()

                    total_top_3_len = len(top_3_percent_values)
                    total_bottom_3_len = len(bottom_3_percent_values)
                   
                    # Extract longitude and latitude columns
                    min_longitude = self.filtered_data['Longitude'].min()
                    max_longitude = self.filtered_data['Longitude'].max()
                    min_latitude = self.filtered_data['Latitude'].min()
                    max_latitude = self.filtered_data['Latitude'].max()

                    # Calculate the area covered (approximate, assuming flat plane)
                    # Convert lat/lon degrees to meters
                    lat_diff_meters = (max_latitude - min_latitude) * 111320  # 1 degree latitude ≈ 111.32 km
                    lon_diff_meters = (max_longitude - min_longitude) * (111320 * np.cos(np.radians((max_latitude + min_latitude) / 2)))

                    # Calculate the area covered in square meters
                    area_covered = abs(lat_diff_meters * lon_diff_meters)
                    #area_covered = abs((max_longitude - min_longitude) * (max_latitude - min_latitude))

                    #if 'Magnetic_Values' in self.filtered_data.columns and self.sampling_frequency is not None:
                    #                    # Frequency Analysis using FFT
                    #    N = len(self.filtered_data['Magnetic_Values'])
                    #    T = 1.0 / self.sampling_frequency
                    #    yf = fft(self.filtered_data['Magnetic_Values'])
                    #    xf = fftfreq(N, T)[:N//2] # Get positive frequencies
#
                    #    # Identify dominant frequencies (peaks in the spectrum)
                    #    dominant_frequencies_indices = np.argsort(np.abs(yf[:N//2]))[::-1][:10] # Top 10 peaks
                    #    dominant_frequencies = xf[dominant_frequencies_indices]

                    # Format the text to display
                    stats_text = (
                        f"Total Number of Columns = {total_columns}\n"
                        f"Total Number of Rows (mag_values) = {total_rows}\n"
                        f"Total Top 3% of Mag values are > {total_top_3} (count{total_top_3_len})\n"
                        f"Total Bottom 3% of Mag values are < {total_bottom_3} (count{total_bottom_3_len})\n"
                        f"Area Covered = {area_covered:.2f} Square Meter\n"
                        f"Coordinates = ( Min Longitude = {min_longitude:.4f}, Max Longitude = {max_longitude:.4f}, Min Latitude = {min_latitude:.4f}, Max Latitude = {max_latitude:.4f} )"
                        #f"Estimated Sampling Frequency: {self.sampling_frequency:.2f} Hz"
                        #f"Dominant Frequencies (Hz):{dominant_frequencies}"
                    )

                    # Plot the text in the figure
                    ax = self.canvas.figure.add_subplot(111)
                    ax.text(0.1, 0.5, stats_text, fontsize=14, verticalalignment='center', horizontalalignment='left',
                            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8))

                    # Remove axes for cleaner display
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)

                    # Draw the canvas
                    self.canvas.draw()
                elif analysis_type1 == "CDF Analysis":
                    self.plot_cdf()
                
                elif analysis_type1 == "Frequency Spectrum":
                    self.plot_frequency_spectrum()

                elif analysis_type1 == "Power Spectral Density":
                    self.plot_PSD()

                elif analysis_type1 == "Time Series":
                        self.plot_time_series()
                else:
                    QMessageBox.warning(self, "Error", "The CSV file does not have the required columns.")
            else:
                QMessageBox.warning(self, "Error", "No data loaded. Please open a CSV file first.")

    def plot_cdf(self):
        # Clear the previous plot
        self.canvas.figure.clear()
        self.filtered_data = self.extract_selected_data()

        # Plot the CDF
        ax = self.canvas.figure.add_subplot(111)
        # Extract 'Mag' data
        mag_values = self.filtered_data['Magnetic_Values'].dropna().sort_values()
        # Compute CDF
        cdf = np.arange(1, len(mag_values) + 1) / len(mag_values)
        # Calculate statistics
        mean_mag = np.mean(mag_values)
        median_mag = np.median(mag_values)
        std_mag = np.std(mag_values)
        max_mag = np.max(mag_values)
        min_mag = np.min(mag_values)
        
        ax.plot(mag_values, cdf, marker='.', linestyle='none', label='Empirical CDF')

        # Plot key statistics
        ax.axvline(mean_mag, color='r', linestyle='--', label=f'Mean: {mean_mag:.2f}')
        ax.axvline(median_mag, color='g', linestyle='--', label=f'Median: {median_mag:.2f}')
        ax.axvline(max_mag, color='b', linestyle='--', label=f'Max: {max_mag:.2f}')
        ax.axvline(min_mag, color='purple', linestyle='--', label=f'Min: {min_mag:.2f}')
        #plt.axvline(total_count, label=f'Total Count: {total_count:.2f}')
        #plt.axvline(total_top_5, label=f'Top 5% Count: {total_top_5:.2f}')
        #plt.axvline(total_bottom_5, label=f'Bottom 5% Count: {total_bottom_5:.2f}')
        ax.fill_betweenx(cdf, mean_mag - std_mag, mean_mag + std_mag, color='orange', alpha=0.3, label=f'Standard Deviation: {std_mag:.2f}')
        
        # Labels and legend
        ax.set_xlabel("Magnitude (Mag)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("CDF Analysis of Magnitude Data")
        ax.legend()
        ax.grid()

        # Refresh the canvas
        self.canvas.draw()

    def plot_frequency_spectrum(self):
        """Plots the Frequency Spectrum"""
        self.filtered_data = self.extract_selected_data()

        if 'Date' in self.filtered_data.columns and 'Time' in self.filtered_data.columns:
            try:
                            # **ADJUSTED FORMAT STRING TO MATCH YOUR DATA**
                datetime_format = '%Y/%m/%d %H:%M:%S.%f'
                self.filtered_data['DateTime'] = pd.to_datetime(self.filtered_data['Date'] + ' ' + self.filtered_data['Time'], format=datetime_format, errors='coerce')

                self.filtered_data.dropna(subset=['DateTime'], inplace=True)

                if not self.filtered_data.empty:
                    self.filtered_data['Time_seconds'] = (self.filtered_data['DateTime'] - self.filtered_data['DateTime'].iloc[0]).dt.total_seconds()
                    time = self.filtered_data['Time_seconds'].values

                    if len(time) > 1:
                        time_diff = np.diff(time)
                        sampling_frequency = 1.0 / np.mean(time_diff)

                    else:
                        print("Error: Not enough time data to estimate sampling frequency.")
                        sampling_frequency = None

                    if sampling_frequency is not None:
                                        # Frequency Analysis using FFT
                        N = len(self.filtered_data['Magnetic_Values'])
                        T = 1.0 / sampling_frequency
                        yf = fft(self.filtered_data['Magnetic_Values'])
                        xf = fftfreq(N, T)[:N//2] # Get positive frequencies

                        plt.figure(figsize=(12, 6))
                        plt.plot(xf, np.abs(yf[:N//2]) * 2 / N, label="Amplitude Spectrum", color='blue') # Plot amplitude spectrum
                        plt.xlabel("Frequency (Hz)")
                        plt.ylabel("Amplitude")
                        plt.title("Frequency Spectrum of Mag")
                        plt.grid(True)
                        plt.show()

                else:
                    print("Error: All rows had issues converting to datetime.")
                    sampling_frequency = None

            except Exception as e:
                print(f"Error combining 'Date' and 'Time' or converting to datetime: {e}")
                sampling_frequency = None

        else:
            print("Error: 'Date' or 'Time' column not found.")
            sampling_frequency = None


    def plot_PSD(self):
        #self.filtered_data = self.extract_selected_data()

        self.filtered_data = self.extract_selected_data()

        if 'Date' in self.filtered_data.columns and 'Time' in self.filtered_data.columns:
            try:
                            # **ADJUSTED FORMAT STRING TO MATCH YOUR DATA**
                datetime_format = '%Y/%m/%d %H:%M:%S.%f'
                self.filtered_data['DateTime_PSD'] = pd.to_datetime(self.filtered_data['Date'] + ' ' + self.filtered_data['Time'], format=datetime_format, errors='coerce')

                self.filtered_data.dropna(subset=['DateTime_PSD'], inplace=True)

                if not self.filtered_data.empty:
                    self.filtered_data['Time_seconds_PSD'] = (self.filtered_data['DateTime_PSD'] - self.filtered_data['DateTime_PSD'].iloc[0]).dt.total_seconds()
                    time = self.filtered_data['Time_seconds_PSD'].values

                if len(time) > 1:
                    time_diff = np.diff(time)
                    sampling_frequency = 1.0 / np.mean(time_diff)

                else:
                    print("Error: Not enough time data to estimate sampling frequency.")
                    sampling_frequency = None

                if 'Magnetic_Values' in self.filtered_data.columns and sampling_frequency is not None:
                                                            # Frequency Analysis using FFT
                    N = len(self.filtered_data['Magnetic_Values'])
                    T = 1.0 / sampling_frequency
                    yf = fft(self.filtered_data['Magnetic_Values'])
                    xf = fftfreq(N, T)[:N//2] # Get positive frequencies

                                # Identify dominant frequencies (peaks in the spectrum)
                    dominant_frequencies_indices = np.argsort(np.abs(yf[:N//2]))[::-1][:10] # Top 10 peaks
                    dominant_frequencies = xf[dominant_frequencies_indices]

                                # Noise Analysis using Welch's Method (PSD)
                    frequencies, power_spectral_density = welch(self.filtered_data['Magnetic_Values'], fs=sampling_frequency, nperseg=1024) # Adjust nperseg

                                # Estimate Noise Level (e.g., average PSD over a frequency range)
                    noise_frequency_range = (10, sampling_frequency / 2) # Example: above 10 Hz
                    noise_indices = np.where((frequencies >= noise_frequency_range[0]) & (frequencies <= noise_frequency_range[1]))
                    if noise_indices[0].size > 0:
                        average_noise_psd = np.mean(power_spectral_density[noise_indices])
                        print(f"Estimated Average Noise PSD (in the range {noise_frequency_range[0]}-{noise_frequency_range[1]} Hz): {average_noise_psd:.2e}")
                    else:
                        print("Could not estimate noise PSD in the specified range.")
                    
                    if frequencies is not None and power_spectral_density is not None:
                    # Estimate Noise Level (as before)
                        noise_frequency_range = (10, sampling_frequency / 2) # Example
                        noise_indices = np.where((frequencies >= noise_frequency_range[0]) & (frequencies <= noise_frequency_range[1]))
                        if noise_indices[0].size > 0:
                            average_noise_psd_db = np.mean(10 * np.log10(power_spectral_density[noise_indices]))

                            plt.figure(figsize=(12, 6))
                            plt.plot(frequencies, 10 * np.log10(power_spectral_density), label='PSD of Mag')
                            plt.axhline(y=average_noise_psd_db, color='r', linestyle='--', label=f'Estimated Noise Level ({average_noise_psd_db:.2f} dB/Hz)')
                            plt.xlabel("Frequency (Hz)")
                            plt.ylabel("Power Spectral Density (dB/Hz)")
                            plt.title("Power Spectral Density of Mag (Welch's Method)")
                            plt.grid(True)
                            plt.legend()
                            plt.show()
                        else:
                            print("Could not estimate noise PSD in the specified range.")

                    else:
                        print("Error: 'DateTime' or 'Mag' column not found in the DataFrame.")

            except Exception as e:
                print(f"Error combining 'Date' and 'Time' or converting to datetime: {e}")
                sampling_frequency = None

        else:
            print("Error: 'Date' or 'Time' column not found.")
            sampling_frequency = None
        

        

        

    def plot_time_series(self):
        """Plots the time series of Mag vs. Time and enables hover functionality."""
        self.canvas.figure.clear()  # Clear the previous plot
        ax = self.canvas.figure.add_subplot(111)
        self.filtered_data = self.extract_selected_data()

        # Ensure correct column names
        required_columns = {'Time', 'Magnetic_Values', 'Latitude', 'Longitude'}
        if not required_columns.issubset(self.filtered_data.columns):
            raise ValueError(f"CSV must contain {required_columns} columns.")

        # Convert Time to datetime format
        self.filtered_data['Time'] = pd.to_datetime(self.filtered_data['Time'])
        # Plot the time series
        # Determine the min and max values of Mag to ensure consistency
        vmin = self.filtered_data['Magnetic_Values'].min()
        vmax = self.filtered_data['Magnetic_Values'].max()
        self.scatter = ax.scatter(self.filtered_data['Time'], self.filtered_data['Magnetic_Values'],s=1,  c=self.filtered_data['Magnetic_Values'], 
                                  cmap="viridis", alpha=0.8, vmin=vmin, vmax=vmax, label='Mag Values')

        # Labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Mag')
        ax.set_title('Time Series of Mag Values')

        # Rotate x-axis labels for readability
        ax.tick_params(axis='x', rotation=45)
        cbar = self.canvas.figure.colorbar(self.scatter,  orientation='vertical')
        cbar.set_label('Magnitude (Mag)')  # Label for the colorbar
        cbar.mappable.set_clim(vmin, vmax)  # Ensure colorbar limits match scatter
        # Adjust colorbar tick labels for better readability
        cbar.set_ticks(np.linspace(vmin, vmax, num=6))  # Set 6 evenly spaced ticks
        # Add a text annotation for hover information
        self.hover_text = ax.annotate("", xy=(0, 0), xytext=(-40, 20),
                                           textcoords="offset points", fontsize=1,
                                           bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

        self.hover_text.set_visible(True)  # Hide by default

        # Connect hover event
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Draw the updated canvas
        self.canvas.draw()

    def on_hover(self, event):
        """Handles hover event to show Latitude & Longitude."""
        #if event.inaxes != ax:
            #return  # Ignore events outside the plot
        ax = self.canvas.figure.add_subplot(111)
        if event.inaxes != ax or self.scatter is None:
            return  # Ignore events outside the plot or if scatter plot is not set

        # Ensure event data is valid
        if event.xdata is None or event.ydata is None:
            self.hover_text.set_visible(False)
            return  # Ignore invalid events

        # Find the closest point to the cursor
        time_numeric = pd.to_datetime(self.data['Time']).view(np.int64) // 10**9  # Convert nanoseconds to seconds
        event_x = float(event.xdata)   # Event x-data is already in float (seconds)
        distances = np.abs(time_numeric - event_x)
        min_index = np.argmin(distances)

        # Retrieve the closest point's data
        if 0 <= min_index < len(self.data):
            lat = self.data['Latitude'].iloc[min_index]
            lon = self.data['Longitude'].iloc[min_index]
        #mag = self.data['Mag'].iloc[min_index]

        # Update annotation position and text
            self.hover_text.set_text(f"Lat: {lat:.4f}, Lon: {lon:.4f}")
            self.hover_text.set_position((event.x, event.y))
            self.hover_text.set_visible(True)
        else:
            self.hover_text.set_visible(False)  # Hide if no valid index found

        # Update canvas
        self.canvas.draw()

    def update_inputs(self):
        """ Update sliders based on selected method """
        method = self.anomaly_detection_method_dropdown.currentText()
        self.description_box.setHtml(self.anomaly_detection_method_descriptions[method])  # Update description

        self.param1_input.clear()
        self.param2_input.clear()

        if method == "Select Method":
            self.param1_label.hide()
            self.param1_input.hide()
            self.param2_label.hide()
            self.param2_input.hide()
            self.submit_button.hide()
            self.description_box.hide()
        else:
            self.param1_label.show()
            self.param1_input.show()
            self.param2_label.show()
            self.param2_input.show()
            self.submit_button.show()
            self.description_box.show()

        if method == "Threshold Detection":
            self.filtered_data = self.extract_selected_data()
            mag_values = self.filtered_data['Magnetic_Values'].dropna().sort_values()
            mean_mag = np.mean(mag_values)
            mean_mag_int = int(mean_mag)  # Convert to integer
            max_mag = np.max(mag_values)
            min_mag = np.min(mag_values)
            self.param1_label.setText("Low Threshold: {} to {}".format(min_mag, mean_mag_int))
            self.param2_label.setText("High Threshold: {} to {}".format(mean_mag_int, max_mag))
            self.param1_input.setPlaceholderText("e.g., 49000")
            self.param2_input.setPlaceholderText("e.g., 51000")
            self.param2_label.show()
            self.param2_input.show()

        elif method == "Quantile Detection":
            self.param1_label.setText("Low Quantile (0.01-0.50)")
            self.param2_label.setText("High Quantile (0.51-0.99)")
            self.param1_input.setPlaceholderText("e.g., 0.01")
            self.param2_input.setPlaceholderText("e.g., 0.99")
            self.param2_label.show()
            self.param2_input.show()

        elif method == "Inter Quartile Range Detection":
            self.param1_label.setText("IQR Multiplier (1.0-20.0)")
            self.param1_input.setPlaceholderText("e.g., 1.5")
            self.param2_label.hide()
            self.param2_input.hide()

        elif method == "ESD Detection":
            self.param1_label.setText("Alpha (0-1)")
            self.param1_input.setPlaceholderText("e.g., 0.05")
            self.param2_label.hide()
            self.param2_input.hide()

        elif method == "PersistAD":
            self.param1_label.setText("C Value (1-10)")
            self.param1_input.setPlaceholderText("e.g., 3")
            self.param2_label.hide()
            self.param2_input.hide()

        elif method == "LevelShiftAD":
            self.param1_label.setText("C Value (1-10)")
            self.param2_label.setText("Window Size (1-20)")
            self.param1_input.setPlaceholderText("e.g., 6")
            self.param2_input.setPlaceholderText("e.g., 5")
            self.param2_label.show()
            self.param2_input.show()

        self.anomaly_detection()

    def anomaly_detection(self):
        """Perform anomaly detection on the standardized data with zoom and save options."""

        # Ensure data is loaded
        if not hasattr(self, "data") or self.data is None:
            print("Error: No data loaded. Please open a file first.")
            return

        # Extract and standardize selected data
        self.filtered_data = self.extract_selected_data()

        if self.filtered_data is None or self.filtered_data.empty:
            print("Error: No data selected for anomaly detection.")
            return

        # Use standardized column names directly
        time_col = "Time"
        mag_col = "Magnetic_Values"

        if time_col not in self.filtered_data.columns or mag_col not in self.filtered_data.columns:
            print(f"Error: Columns '{time_col}' or '{mag_col}' not found in filtered data.")
            print(f"Available columns: {self.filtered_data.columns.tolist()}")
            return

        # Set the Time column as index and select the MagneticValues column
        self.filtered_data["Time"] = pd.to_datetime(self.filtered_data["Time"])
        data = self.filtered_data.set_index("Time")["Magnetic_Values"]

        print(f"Data Ready for Anomaly Detection:\n{data.head()}")  # Debugging output

        # Get input values safely
        try:
            param1 = float(self.param1_input.text()) if self.param1_input.text() else None
            param2 = float(self.param2_input.text()) if self.param2_input.text() else None
        except ValueError:
            return  # Ignore invalid input

        try:
            # Clear previous figure
            self.canvas.figure.clear()
            #ax = self.canvas.figure.subplot()
            analysis_type2 = self.anomaly_detection_method_dropdown.currentText()

            # Apply the selected anomaly detection method
            if analysis_type2 == "Threshold Detection" and param1 is not None and param2 is not None:
                threshold_detector = ThresholdAD(low=param1, high=param2)
                anomalies = threshold_detector.detect(data)

            elif analysis_type2 == "Quantile Detection" and param1 is not None and param2 is not None:
                quantile_detector = QuantileAD(low=param1, high=param2)
                anomalies = quantile_detector.fit_detect(data)

            elif analysis_type2 == "Inter Quartile Range Detection" and param1 is not None:
                
                iqr = InterQuartileRangeAD(param1)
                anomalies = iqr.fit_detect(data)

            elif analysis_type2 == "ESD Detection" and param1 is not None:
                
                data = validate_series(data)
                esd = GeneralizedESDTestAD(alpha=param1)
                anomalies = esd.fit_detect(data)

            elif analysis_type2 == "PersistAD" and param1 is not None:
                
                data = validate_series(data)
                persist_ad = PersistAD(c=param1, side="positive")
                anomalies = persist_ad.fit_detect(data)

            elif analysis_type2 == "LevelShiftAD" and param1 is not None and param2 is not None:
                
                data = validate_series(data)
                ls_ad = LevelShiftAD(c=param1, side="both", window=int(param2))
                anomalies = ls_ad.fit_detect(data)

            else:
                print("Error: Invalid anomaly detection method selected.")
                return

            # Plot anomalies using Matplotlib
            #self.plot_anomalies(data, anomalies)
            #plt.style.use("ggplot")
            plot(data, anomaly = anomalies, anomaly_color="red", anomaly_tag = "marker")
            plt.show()


        except Exception as e:
            print("Error in anomaly detection:", str(e))
                  
    def processing_tool(self):

        analysis_type4 = self.tool_box.currentText()
        self.param1_label.hide()
        self.param1_input.hide()
        self.param2_label.hide()
        self.param2_input.hide()
        self.submit_button.hide()
        self.description_box.hide()

        # Check if filtered_data exists and has more than the initial columns
        # This suggests that some processing (like LPF) has already been applied.
        initial_columns = ["Longitude", "Latitude", "Magnetic_Values", "Altitude", "Time", "Date"]
        processed_before = hasattr(self, 'filtered_data') and self.filtered_data is not None and len(self.filtered_data.columns) > len(set(initial_columns) & set(self.filtered_data.columns))

        if not processed_before:
            self.filtered_data = self.extract_selected_data()

        if self.filtered_data is not None:
            if len(self.filtered_data.columns) >= 1:
                # Extract the selected analysis type
                self.description_box.hide()

                if analysis_type4 == "Low Pass Filter":
                    columns = self.filtered_data.columns  # Get column names
                    self.lpf_dialog = QDialog(self)
                    self.lpf_dialog.setWindowTitle("Low Pass Filter")
                    self.lpf_dialog.setGeometry(300, 300, 400, 200)

                    # Labels and Dropdowns
                    self.channel_label = QLabel("Channel to filter:")
                    self.channel_dropdown = QComboBox()
                    if not self.filtered_data.empty:
                        self.channel_dropdown.addItems(self.filtered_data.columns)
                    else:
                        self.channel_dropdown.addItem("No data available")
                        self.channel_dropdown.setEnabled(False)

                    self.output_label = QLabel("Output Channel:")
                    self.output_name = QLineEdit()
                    self.output_name.setPlaceholderText("Auto-generated name")
                    self.output_name.setReadOnly(True)

                    self.cutoff_label = QLabel("Cutoff Wavelength (fiducials):")
                    self.cutoff_input = QLineEdit()
                    self.cutoff_input.setValidator(QDoubleValidator(0, 100000, 2))
                    self.cutoff_input.setPlaceholderText("Nil (Default)")

                    # Buttons
                    self.ok_button = QPushButton("OK")
                    self.cancel_button = QPushButton("Cancel")

                    self.ok_button.clicked.connect(self.apply_filter)
                    self.cancel_button.clicked.connect(self.close)

                    # Layout
                    layout = QVBoxLayout()
                    layout.addWidget(self.channel_label)
                    layout.addWidget(self.channel_dropdown)
                    layout.addWidget(self.output_label)
                    layout.addWidget(self.output_name)
                    layout.addWidget(self.cutoff_label)
                    layout.addWidget(self.cutoff_input)

                    button_layout = QHBoxLayout()
                    button_layout.addWidget(self.ok_button)
                    button_layout.addWidget(self.cancel_button)
                    layout.addLayout(button_layout)

                    # Auto-generate output name
                    self.channel_dropdown.currentIndexChanged.connect(self.update_output_name_lpf)

                    self.lpf_dialog.setLayout(layout)
                    self.lpf_dialog.exec_()

                    
                    
                elif analysis_type4 == "Time Lag Correction":
                    self.lag_dialog = QDialog(self)
                    self.lag_dialog.setWindowTitle("Time Lag Correction")
                    self.lag_dialog.setGeometry(300, 300, 400, 200)

                    # Labels and Dropdowns
                    self.channel_label = QLabel("Channel for Lag Correction:")
                    self.channel_dropdown = QComboBox()
                    if not self.filtered_data.empty:
                        self.channel_dropdown.addItems(self.filtered_data.columns)
                    else:
                        self.channel_dropdown.addItem("No data available")
                        self.channel_dropdown.setEnabled(False)

                    self.output_label = QLabel("Output Channel:")
                    self.output_name = QLineEdit()
                    self.output_name.setPlaceholderText("Auto-generated name")
                    self.output_name.setReadOnly(True)

                    self.lag_label = QLabel("Lag (fiducials):")
                    self.lag_input = QLineEdit()
                    self.lag_input.setValidator(QDoubleValidator(-10000, 100000, 2))
                    self.lag_input.setPlaceholderText("Enter lag (e.g., 100)")

                    # Buttons
                    self.ok_button = QPushButton("OK")
                    self.cancel_button = QPushButton("Cancel")

                    self.ok_button.clicked.connect(self.apply_time_lag)
                    self.cancel_button.clicked.connect(self.close)

                    # Layout
                    layout = QVBoxLayout()
                    layout.addWidget(self.channel_label)
                    layout.addWidget(self.channel_dropdown)
                    layout.addWidget(self.output_label)
                    layout.addWidget(self.output_name)
                    layout.addWidget(self.lag_label)
                    layout.addWidget(self.lag_input)

                    button_layout = QHBoxLayout()
                    button_layout.addWidget(self.ok_button)
                    button_layout.addWidget(self.cancel_button)
                    layout.addLayout(button_layout)

                    # Auto-generate output name
                    self.channel_dropdown.currentIndexChanged.connect(self.update_output_name_lag)

                    self.lag_dialog.setLayout(layout)
                    self.lag_dialog.exec_()
                   

                elif analysis_type4 == "Residual Mag Anomaly":
                    self.res_anomaly_dialog = QDialog(self)
                    self.res_anomaly_dialog.setWindowTitle("Residual Mag Anomaly")
                    self.res_anomaly_dialog.setGeometry(300, 300, 400, 200)

                    # Labels and Dropdowns
                    self.channel_label = QLabel("Channel for Residual Anomaly:")
                    self.channel_dropdown = QComboBox()
                    if not self.filtered_data.empty:
                        self.channel_dropdown.addItems(self.filtered_data.columns)
                    else:
                        self.channel_dropdown.addItem("No data available")
                        self.channel_dropdown.setEnabled(False)

                    self.output_label = QLabel("Output Channel:")
                    self.output_name = QLineEdit()
                    self.output_name.setPlaceholderText("Auto-generated name")
                    self.output_name.setReadOnly(True)

                    self.res_anomaly_label = QLabel("Rolling Median Window:")
                    self.res_anomaly_input = QLineEdit()
                    self.res_anomaly_input.setValidator(QDoubleValidator(0, 100000, 2))
                    self.res_anomaly_input.setPlaceholderText("Enter Window (e.g., 20000)")

                    # Buttons
                    self.ok_button = QPushButton("OK")
                    self.cancel_button = QPushButton("Cancel")

                    self.ok_button.clicked.connect(self.apply_residual_anomaly)
                    self.cancel_button.clicked.connect(self.close)

                    # Layout
                    layout = QVBoxLayout()
                    layout.addWidget(self.channel_label)
                    layout.addWidget(self.channel_dropdown)
                    layout.addWidget(self.output_label)
                    layout.addWidget(self.output_name)
                    layout.addWidget(self.res_anomaly_label)
                    layout.addWidget(self.res_anomaly_input)

                    button_layout = QHBoxLayout()
                    button_layout.addWidget(self.ok_button)
                    button_layout.addWidget(self.cancel_button)
                    layout.addLayout(button_layout)

                    # Auto-generate output name
                    self.channel_dropdown.currentIndexChanged.connect(self.update_output_name_res_anomaly)

                    self.res_anomaly_dialog.setLayout(layout)
                    self.res_anomaly_dialog.exec_()
                    

                elif analysis_type4 == "Analytic Signal Generation":
                    print("Error in visualisation method:", str(e))

            else:
                print("Error in Processing Data:", str(e))

    def update_output_name_lpf(self):
        selected_channel = self.channel_dropdown.currentText()
        self.output_name.setText(f"{selected_channel}_LPF")

    def update_output_name_lag(self):
        selected_channel = self.channel_dropdown.currentText()
        self.output_name.setText(f"{selected_channel}_LagCorrected")

    def update_output_name_res_anomaly(self):
        self.output_name.setText(f"Residual Anomaly (nT)")


    def apply_filter(self):
        selected_channel = self.channel_dropdown.currentText()
        output_column = f"{selected_channel}_LPF"
        cutoff = self.cutoff_input.text()

        if cutoff == "" or cutoff.lower() == "nil":
            cutoff = 0.1  # Default cutoff frequency
        else:
            try:
                cutoff = float(cutoff)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for cutoff frequency.")
                return

        # Check if filtered_data is empty
        if self.filtered_data.empty:
            QMessageBox.warning(self, "No Data", "No data available to filter.")
            return

        # Remove previous LPF column if it exists
        if output_column in self.filtered_data.columns:
            self.filtered_data.drop(columns=[output_column], inplace=True)

        # Apply Low Pass Filter
        # Gaussian filter
        
        self.filtered_data[output_column] = gaussian_filter1d(self.filtered_data[selected_channel], sigma = cutoff )
        self.lpf_dialog.close()   

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(self.filtered_data[selected_channel], label="Original", color='blue')
        plt.plot(self.filtered_data[output_column], label="Low Pass Filtered", color='red')
        plt.xlabel("Index")
        plt.ylabel("Magnitude")
        plt.title(f"Low Pass Filtering: {selected_channel} -> {output_column}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()    

    def apply_time_lag(self):
        selected_channel = self.channel_dropdown.currentText()
        output_column = f"{selected_channel}_Lag"
        lag_text = self.lag_input.text()

        if lag_text == "":
            QMessageBox.warning(self.dialog, "Invalid Input", "Please enter a valid lag value.")
            return

        try:
            lag = int(float(lag_text))
        except ValueError:
            QMessageBox.warning(self.dialog, "Invalid Input", "Lag must be a number.")
            return

        # Check if filtered_data is empty
        if self.filtered_data.empty:
            QMessageBox.warning(self, "No Data", "No data available for lag correction.")
            return

        # Remove previous LPF column if it exists
        if output_column in self.filtered_data.columns:
            self.filtered_data.drop(columns=[output_column], inplace=True)

        self.filtered_data[output_column] = self.filtered_data[selected_channel].shift(lag).ffill()
        self.lag_dialog.close()
        # Plot result
        plt.figure(figsize=(14, 5))
        plt.plot(self.filtered_data[selected_channel], label=f"Original ({selected_channel})", color='blue')
        plt.plot(self.filtered_data[output_column], label=f"Lag Corrected ({output_column})", color='red')
        plt.title("Time Lag Correction")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def apply_residual_anomaly(self):
        selected_channel = self.channel_dropdown.currentText()
        output_column = f"Residual Anomaly (nT)"
        window_size = self.res_anomaly_input.text()

        if self.filtered_data.empty:
            QMessageBox.warning(self.dialog, "No Data", "No data available to process.")
            return

        if output_column in self.filtered_data.columns:
            self.filtered_data.drop(columns=[output_column], inplace=True)

        # Calculate rolling median
        rolling_window = int(window_size)
        rolling_median = self.filtered_data[selected_channel].rolling(window=rolling_window, center=True).median()

        # Subtract rolling median to get residual anomaly
        self.filtered_data[output_column] = self.filtered_data[selected_channel] - rolling_median

        self.res_anomaly_dialog.close()

        # Plot result
        plt.figure(figsize=(14, 5))
        #plt.plot(self.filtered_data[selected_channel], label=f"Lag Corrected ({selected_channel})", color='green')
        plt.plot(self.filtered_data[output_column], label=f"Residual Anomaly ({output_column})", color='red')
        plt.title("Residual Magnetic Anomaly")
        plt.xlabel("Index")
        plt.ylabel("Magnetic Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def update_inputs_vis(self):
        """ Update sliders based on selected method """
        method = self.filter_box3.currentText()
        #self.description_box.setHtml(self.anomaly_detection_method_descriptions[method])  # Update description
        self.description_box.hide()
        self.param3_input.clear()
        self.param4_input.clear()
        self.param5_input.clear()

        if method == "Select Method":
            self.param3_label.hide()
            self.param3_input.hide()
            self.param4_label.hide()
            self.param4_input.hide()
            self.param5_label.hide()
            self.param5_input.hide()
            self.submit_button_vis.hide()
            
        else:
            self.param3_label.show()
            self.param3_input.show()
            self.param4_label.show()
            self.param4_input.show()
            self.param5_label.show()
            self.param5_input.show()
            self.submit_button_vis.show()
            

        if method == "Anomaly Data Points Plot":
            self.param3_label.hide()
            self.param3_input.hide()
            self.param4_label.hide()
            self.param4_input.hide()
            self.param5_label.hide()
            self.param5_input.hide()
            self.submit_button_vis.hide()
            
        elif method == "Residual Anomaly Plot":
            self.param3_input.setPlaceholderText("e.g., 0.5")
            self.param4_input.setPlaceholderText("e.g., 2")
            self.param5_input.setPlaceholderText("e.g., 0.5")
            #self.param2_label.show()
            #self.param2_input.show()

        elif method == "Analytic Signal (AS) Plot":
            self.param3_input.setPlaceholderText("e.g., 0.5")
            self.param4_input.setPlaceholderText("e.g., 2")
            self.param5_input.setPlaceholderText("e.g., 0.5")

        elif method == "AS (Grided & Countoured)":
            self.param3_input.setPlaceholderText("e.g., 0.5")
            self.param4_input.setPlaceholderText("e.g., 2")
            self.param5_input.setPlaceholderText("e.g., 0.5")

        elif method == "AS 3D Plot":
            self.param3_input.setPlaceholderText("e.g., 0.5")
            self.param4_input.setPlaceholderText("e.g., 2")
            self.param5_input.setPlaceholderText("e.g., 0.5")

        elif method == "AS on Map":
            self.param3_input.setPlaceholderText("e.g., 0.5")
            self.param4_input.setPlaceholderText("e.g., 2")
            self.param5_input.setPlaceholderText("e.g., 0.5")

        self.vis_method()

    def vis_method(self):
        #self.filtered_data = self.extract_selected_data()
        analysis_type3 = self.filter_box3.currentText()
        self.param1_label.hide()
        self.param1_input.hide()
        self.param2_label.hide()
        self.param2_input.hide()
        self.submit_button.hide()
        self.description_box.hide()

        # Get input values safely
        try:
            param3 = float(self.param3_input.text()) if self.param3_input.text() else None
            param4 = float(self.param4_input.text()) if self.param4_input.text() else None
            param5 = float(self.param5_input.text()) if self.param5_input.text() else None
        except ValueError:
            return  # Ignore invalid input


        if self.filtered_data is not None:
            if len(self.filtered_data.columns) >= 1:
                # Extract the selected analysis type
                self.description_box.hide()

                #lon = self.filtered_data["Longitude"]
                #lat = self.filtered_data["Latitude"]
                #mag = self.filtered_data["Magnetic_Values"]
                #mag_anomaly = self.filtered_data["Residual Anomaly (nT)"]

                # Drop NaNs for plotting
                df_valid = self.filtered_data.dropna(subset=["Latitude", "Longitude", "Magnetic_Values", "Residual Anomaly (nT)"])

                # Create grid for interpolation
                num_points_x, num_points_y = 500, 500
                grid_x = np.linspace(df_valid["Longitude"].min(), df_valid["Longitude"].max(), num_points_x)
                grid_y = np.linspace(df_valid["Latitude"].min(), df_valid["Latitude"].max(), num_points_y)
                grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
                grid_z_residual = griddata(
                    (df_valid["Longitude"], df_valid["Latitude"]),
                    df_valid["Residual Anomaly (nT)"],
                    (grid_x_mesh, grid_y_mesh),
                    method="cubic"  # Try cubic or other methods
                )
                          
                if analysis_type3 == "Anomaly Data Points Plot": 

                    # Convert to GeoDataFrame (for mapping)
                    gdf = gpd.GeoDataFrame(df_valid, geometry=gpd.points_from_xy(df_valid["Longitude"], df_valid["Latitude"]), crs="EPSG:4326")

                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6), dpi=125, subplot_kw={"projection": ccrs.PlateCarree()})

                    # Set map extent
                    ax.set_extent([df_valid["Longitude"].min(), df_valid["Longitude"].max(), df_valid["Latitude"].min(), df_valid["Latitude"].max()], crs=ccrs.PlateCarree())

                    # Add coastlines and features
                    ax.add_feature(cfeature.COASTLINE)
                    ax.add_feature(cfeature.BORDERS, linestyle=':')
                    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='white')
                    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

                    # Scatter plot for points
                    sc = ax.scatter(df_valid["Longitude"], df_valid["Latitude"], c=df_valid["Magnetic_Values"], cmap="coolwarm", s=10, transform=ccrs.PlateCarree())

                    # Add colorbar
                    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
                    cbar.set_label("Magnetic Anomaly (nT)")

                    # Add grid lines
                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
                    gl.top_labels = False  # Remove top latitude values
                    gl.right_labels = False  # Remove right longitude values

                    # Set labels and title
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_title("Magnetic Anomalies as Points Overlaid on a Map")

                    # Show plot
                    plt.show()

                elif analysis_type3 == "Residual Anomaly Plot" and param3 is not None and param4 is not None and param5 is not None:

                    # Smooth residual anomaly
                    grid_z_residual_smooth = gaussian_filter1d(grid_z_residual, sigma=param3, axis=0) # Gaussian smoothing
                    grid_z_residual_smooth = gaussian_filter1d(grid_z_residual_smooth, sigma=param3, axis=1)
                    grid_z_residual_smooth = median_filter(grid_z_residual_smooth, size=int(param4))  # Median filter 

                    # Color normalization (TwoSlopeNorm)
                    mean_val_res = np.nanmean(grid_z_residual_smooth)
                    std_val_res = np.nanstd(grid_z_residual_smooth)
                    print(f"vmin: {(mean_val_res - param5 * std_val_res)}")
                    print(f"vcenter: {(mean_val_res)}")
                    print(f"vmax: {(mean_val_res + param5 * std_val_res)}")
                    norm_res = TwoSlopeNorm(vmin=mean_val_res - param5 * std_val_res, vcenter=mean_val_res, vmax=mean_val_res + param5 * std_val_res) 

                    # Shading
                    ls = LightSource(azdeg=225, altdeg=25)
                    shaded_res = ls.shade(grid_z_residual_smooth, cmap=plt.get_cmap('coolwarm'), norm=norm_res, blend_mode='overlay')

                    # --- Residual Anomaly Plot ---
                    fig, ax = plt.subplots(figsize=(12, 8))

                    im_res = ax.imshow(
                        shaded_res,
                        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                        origin='lower',
                        aspect='auto'
                    )

                    # Filled contours (optional, for comparison)
                    contour_filled_res = ax.contourf(
                        grid_x, grid_y, grid_z_residual_smooth,
                        levels=100,
                        cmap='coolwarm',
                        norm=norm_res,
                        alpha=0.8
                    )

                    # Contour lines
                    contour_lines_res = ax.contour(
                        grid_x, grid_y, grid_z_residual_smooth,
                        levels=10,
                        colors='black',
                        linewidths=0.3,
                        alpha=0.2
                    )
                    ax.clabel(contour_lines_res, inline=True, fontsize=6, fmt='%1.1f')

                    # Colorbar
                    cbar_res = plt.colorbar(contour_filled_res, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
                    cbar_res.set_label("Residual Anomaly (nT)")

                    # Labels and title
                    ax.set_title("Smoothed Residual Magnetic Anomaly")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    plt.grid(True, linestyle='--', alpha=0.2)
                    plt.tight_layout()
                    plt.show()
                
                # --- Analytic Signal Plot ---
                elif analysis_type3 == "Analytic Signal (AS) Plot" and param3 is not None and param4 is not None and param5 is not None:
                    
                    # Calculate analytic signal
                    dx = sobel(grid_z_residual, axis=1, mode='reflect')
                    dy = sobel(grid_z_residual, axis=0, mode='reflect')
                    analytic_signal = np.sqrt(dx**2 + dy**2)

                    # Smooth analytic signal
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal, sigma=param3, axis=0)  # Gaussian smoothing
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal_grid_smooth, sigma=param3, axis=1)
                    analytic_signal_grid_smooth = median_filter(analytic_signal_grid_smooth, size=int(param4))   # Median filter             

                    # --- Handle NaNs ---
                    print(f"Number of NaN values before handling: {np.sum(np.isnan(analytic_signal_grid_smooth))}")
                    #analytic_signal_grid_smooth_as = np.nan_to_num(analytic_signal_grid_smooth, nan=0.1)  # Replace NaNs with 0
                    #print(f"Number of NaN values after handling: {np.sum(np.isnan(analytic_signal_grid_smooth_as))}")
                    # --- End Handle NaNs --

                    fig_as, ax_as = plt.subplots(figsize=(12, 8))

                    # Color normalization (LogNorm)
                    mean_val_as = np.nanmean(analytic_signal_grid_smooth)
                    std_val_as = np.nanstd(analytic_signal_grid_smooth)
                    norm_as = TwoSlopeNorm(vmin=1, vcenter=mean_val_as, vmax=mean_val_as + param5 * std_val_as)

                    # Shading
                    ls_as = LightSource(azdeg=195, altdeg=20)
                    shaded_as = ls_as.shade(analytic_signal_grid_smooth, cmap=plt.get_cmap('coolwarm'), norm=norm_as, blend_mode='soft')

                    im_as = ax_as.imshow(
                        shaded_as,
                        extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                        origin='lower',
                        aspect='auto'
                    )

                    # Filled contours (optional)
                    contour_filled_as = ax_as.contourf(
                        grid_x, grid_y, analytic_signal_grid_smooth,
                        levels=150,
                        cmap='coolwarm',
                        norm=norm_as,
                        alpha=0.9
                    )

                    # Contour lines
                    contour_lines_as = ax_as.contour(
                        grid_x, grid_y, analytic_signal_grid_smooth,
                        levels=10,
                        colors='white',
                        linewidths=0.2,
                        alpha=0.5
                    )
                    ax_as.clabel(contour_lines_as, inline=True, fontsize=8, fmt='%1.1f')

                    # Colorbar
                    cbar_as = plt.colorbar(contour_filled_as, ax=ax_as, orientation='vertical', shrink=0.8, pad=0.02)
                    cbar_as.set_label("Analytic Signal Amplitude (nT/m)")

                    # Labels and title
                    ax_as.set_title("Smoothed Analytic Signal Amplitude")
                    ax_as.set_xlabel("Longitude")
                    ax_as.set_ylabel("Latitude")
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    plt.show()

                elif analysis_type3 == "Residual Anomaly & AS" and param3 is not None and param4 is not None and param5 is not None:
                    
                    # Smooth residual anomaly
                    grid_z_residual_smooth = gaussian_filter1d(grid_z_residual, sigma=param3, axis=0) # Gaussian smoothing
                    grid_z_residual_smooth = gaussian_filter1d(grid_z_residual_smooth, sigma=param3, axis=1)
                    grid_z_residual_smooth = median_filter(grid_z_residual_smooth, size=int(param4))  # Median filter 

                    # Color normalization (TwoSlopeNorm)
                    mean_val_res1 = np.nanmean(grid_z_residual_smooth)
                    std_val_res1 = np.nanstd(grid_z_residual_smooth)
                    norm_res1 = TwoSlopeNorm(vmin=mean_val_res1 - param5 * std_val_res1, 
                                             vcenter=mean_val_res1, 
                                             vmax=mean_val_res1 + param5 * std_val_res1) 

                    # Shading
                    ls = LightSource(azdeg=315, altdeg=25)
                    shaded_res1 = ls.shade(grid_z_residual_smooth, 
                                           cmap=plt.get_cmap('coolwarm'),
                                           norm = norm_res1, 
                                           blend_mode='overlay')

                    gdf = gpd.GeoDataFrame(df_valid, 
                                           geometry=gpd.points_from_xy(df_valid["Longitude"],
                                                                        df_valid["Latitude"]), 
                                                                        crs="EPSG:4326")
                    gdf = gdf.to_crs(epsg=3857)  # Convert to Web Mercator for contextily
                    # Set up the figure for side-by-side plots
                    fig, axes = plt.subplots(1,2, figsize=(12, 6), dpi=125, 
                                             subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True,
                                             sharex=True, sharey=True)

                    # Left Plot - Residual Anomaly (imshow with shading)
                    im0 = axes[0].imshow(shaded_res1,
                                    extent=[grid_x_mesh.min(), grid_x_mesh.max(),
                                            grid_y_mesh.min(), grid_y_mesh.max()],
                                    origin='lower',  # Important for correct orientation
                                    transform=ccrs.PlateCarree())
                                    
                    # Filled contours (optional, for comparison)
                    contour_filled_res = axes[0].contourf(
                        grid_x_mesh, grid_y_mesh, grid_z_residual_smooth,
                        levels=100,
                        cmap='coolwarm',
                        norm=norm_res1,
                        alpha=0.8
                    )

                    # Contour lines
                    contour_lines_res = axes[0].contour(
                        grid_x_mesh, grid_y_mesh, grid_z_residual_smooth,
                        levels=10,
                        colors='black',
                        linewidths=0.3,
                        alpha=0.2
                    )
                    axes[0].clabel(contour_lines_res, inline=True, fontsize=6, fmt='%1.1f')

                    # Colorbar
                    cbar_res = plt.colorbar(contour_filled_res, ax=axes[0], orientation='horizontal', shrink=0.8, pad=0.02)
                    cbar_res.set_label("Residual Anomaly (nT)")

                    axes[0].set_title("Residual Anomaly")
                    axes[0].set_xlabel("Longitude")
                    axes[0].set_ylabel("Latitude")


                    # Add grid lines
                    gl = axes[0].gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
                    gl.top_labels = False  # Remove top latitude values
                    gl.right_labels = False  # Remove right longitude values
                    gl.xlabel_style = {'rotation': 30}  # Rotate longitude labels (bottom/top)

                    #Right Plot - Analytic Signal

                    # Calculate analytic signal
                    dx = sobel(grid_z_residual, axis=1, mode='reflect')
                    dy = sobel(grid_z_residual, axis=0, mode='reflect')
                    analytic_signal = np.sqrt(dx**2 + dy**2)

                    # Smooth analytic signal
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal, sigma=param3, axis=0)  # Gaussian smoothing
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal_grid_smooth, sigma=param3, axis=1)
                    analytic_signal_grid_smooth = median_filter(analytic_signal_grid_smooth, size=int(param4))   # Median filter             

                    # --- Handle NaNs ---
                    print(f"Number of NaN values before handling: {np.sum(np.isnan(analytic_signal_grid_smooth))}")
                    #analytic_signal_grid_smooth_as = np.nan_to_num(analytic_signal_grid_smooth, nan=0.1)  # Replace NaNs with 0
                    #print(f"Number of NaN values after handling: {np.sum(np.isnan(analytic_signal_grid_smooth_as))}")
                    # --- End Handle NaNs --

                    # Color normalization (LogNorm)
                    mean_val_as1 = np.nanmean(analytic_signal_grid_smooth)
                    std_val_as1 = np.nanstd(analytic_signal_grid_smooth)
                    norm_as1 = TwoSlopeNorm(vmin=1, vcenter=mean_val_as1, vmax=mean_val_as1 + param5 * std_val_as1)

                    # Shading
                    ls_as = LightSource(azdeg=315, altdeg=25)
                    shaded_as1 = ls_as.shade(analytic_signal_grid_smooth,
                                             cmap=plt.get_cmap('coolwarm'),
                                             norm=norm_as1, 
                                             blend_mode='soft')
                    
                    # Right Plot - Analytic Signal (imshow with shading)
                    im1 = axes[1].imshow(shaded_as1,
                                    extent=[grid_x_mesh.min(), grid_x_mesh.max(),
                                            grid_y_mesh.min(), grid_y_mesh.max()],
                                    origin='lower',  # Important for correct orientation
                                    transform=ccrs.PlateCarree() )
                                    
                    # Filled contours (optional)
                    contour_filled_as = axes[1].contourf(
                        grid_x_mesh, grid_y_mesh, analytic_signal_grid_smooth,
                        levels=100,
                        cmap='coolwarm',
                        norm=norm_as1,
                        alpha=0.8
                    )

                    # Contour lines
                    contour_lines_as = axes[1].contour(
                        grid_x_mesh, grid_y_mesh, analytic_signal_grid_smooth,
                        levels=10,
                        colors='black',
                        linewidths=0.3,
                        alpha=0.2
                    )
                    axes[1].clabel(contour_lines_as, inline=True, fontsize=6, fmt='%1.1f')

                    # Colorbar
                    cbar_as = plt.colorbar(contour_filled_as, ax=axes[1], orientation='horizontal', shrink=0.8, pad=0.02)
                    cbar_as.set_label("Analytic Signal (nT/m)")

                    axes[1].set_title("Analytic Signal")
                    axes[1].set_xlabel("Longitude")
                    axes[1].set_ylabel("Latitude")

                    
                    # Add grid lines
                    gl1 = axes[1].gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
                    gl1.top_labels = False  # Remove top latitude values
                    gl1.right_labels = True  # Remove right longitude values
                    gl1.left_labels = False
                    gl1.xlabel_style = {'rotation': 30}  # Rotate longitude labels (bottom/top)
                
                    #plt.grid(True, linestyle='--', alpha=0.5)
                    #plt.tight_layout()
                    plt.show()

                elif analysis_type3 == "AS 3D Plot" and param3 is not None and param4 is not None and param5 is not None:
                    
                    # Calculate analytic signal
                    dx = sobel(grid_z_residual, axis=1, mode='reflect')
                    dy = sobel(grid_z_residual, axis=0, mode='reflect')
                    analytic_signal = np.sqrt(dx**2 + dy**2)

                    # Smooth analytic signal
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal, sigma=param3, axis=0)  # Gaussian smoothing
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal_grid_smooth, sigma=param3, axis=1)
                    analytic_signal_grid_smooth = median_filter(analytic_signal_grid_smooth, size=int(param4))   # Median filter  

                    # Color normalization (LogNorm)
                    mean_val_as_3d = np.nanmean(analytic_signal_grid_smooth)
                    std_val_as_3d = np.nanstd(analytic_signal_grid_smooth)
                    norm_as_3d = TwoSlopeNorm(vmin=1, vcenter=mean_val_as_3d, vmax=mean_val_as_3d + param5 * std_val_as_3d)

                    # Shading
                    #ls_as = LightSource(azdeg=195, altdeg=20)
                    #shaded_as = ls_as.shade(analytic_signal_grid_smooth, cmap=plt.get_cmap('coolwarm'), norm=norm_as, blend_mode='soft')           
                    
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ##
                    ### Plot the surface with the normalization
                    surf = ax.plot_surface(grid_x_mesh, grid_y_mesh, analytic_signal_grid_smooth,
                                        cmap=cm.viridis_r,
                                        linewidth=0, antialiased=False, shade=True,
                                        norm=norm_as_3d) # Apply the normalization

                    ### Add a color bar which maps values to colors based on the normalization.
                    fig.colorbar(surf, shrink=0.5, aspect=10, label='Amplitude of Analytic Signal (approx.), nT/m')
                    #
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_zlabel("Amplitude (nT/m)")
                    ax.set_title("3D View of Amplitude of Analytic Signal")
                    #
                    ### Adjust viewing angle (optional)
                    ax.view_init(elev=15, azim=210) # Adjust these angles to get a good perspective
                    ##
                    plt.tight_layout()
                    plt.show() 

                    # Assuming analytic_signal_grid_smooth is your z data
                    #z = np.array(analytic_signal_grid_smooth)

                    # Mask invalid values (like NaN or zero)
                    #mask = np.isnan(z) | (z == 0)  # You can modify this condition as needed
                    #z_masked = np.where(mask, np.nan, z)  # Replace invalid values with NaN

                    ## Plot using Plotly
                    fig = go.Figure(data=go.Surface(
                        x=grid_x_mesh, y=grid_y_mesh, z=analytic_signal_grid_smooth,
                        colorscale='viridis_r',
                        colorbar=dict(title="Elevation (m)")
                    ))
                    #
                    fig.update_layout(
                        title="3D Elevation Map",
                        scene=dict(
                            xaxis_title="Longitude",
                            yaxis_title="Latitude",
                            zaxis_title="Elevation (m)"
                        )
                    )

                    fig.show()

                elif analysis_type3 == "AS on Map" and param3 is not None and param4 is not None and param5 is not None:
                    
                    # Calculate analytic signal
                    dx = sobel(grid_z_residual, axis=1, mode='reflect')
                    dy = sobel(grid_z_residual, axis=0, mode='reflect')
                    analytic_signal = np.sqrt(dx**2 + dy**2)

                    # Smooth analytic signal
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal, sigma=param3, axis=0)  # Gaussian smoothing
                    analytic_signal_grid_smooth = gaussian_filter1d(analytic_signal_grid_smooth, sigma=param3, axis=1)
                    analytic_signal_grid_smooth = median_filter(analytic_signal_grid_smooth, size=int(param4))   # Median filter             

                    # Flatten the gridded data and coordinates for use with Scattermapbox
                    flat_lon = grid_x_mesh.flatten()
                    flat_lat = grid_y_mesh.flatten()
                    flat_z = analytic_signal_grid_smooth.flatten()

                    # Remove NaN values from the data
                    #valid_indices = ~np.isnan(analytic_signal_grid_smooth)
                    #flat_lon = grid_x_mesh[valid_indices]
                    #flat_lat = grid_y_mesh[valid_indices]
                    #flat_z = analytic_signal_grid_smooth[valid_indices]

                    fig = go.Figure(go.Densitymapbox(
                        lon=flat_lon,
                        lat=flat_lat,
                        z=flat_z,
                        radius=8,  # Adjust the radius as needed
                        colorscale='viridis_r',
                        colorbar=dict(title='Analytic Signal (nT/m)'),
                        opacity=1,
                        #zmin=np.nanmin(flat_z), # Optional: Set zmin and zmax for consistent color scaling
                        #zmax=np.nanmax(flat_z)
                    ))

                    fig.update_layout(
                        mapbox_style= "open-street-map", #"stamen-terrain",  # Or "carto-positron", 
                        mapbox_center_lon=np.nanmean(flat_lon),
                        mapbox_center_lat=np.nanmean(flat_lat),
                        mapbox_zoom=18,  # Adjust the zoom level
                        title='Anomalies on Map'
                    )

                    fig.show()


            else:
                print("Error in visualisation method:", str(e))  

    #def generate_heatmap(self):
#
    #    if self.data is not None:
    #        try:
    #            min_mag = data["Mag"].min()
    #            max_mag = data["Mag"].max()
#
    #        except Exception as e:
    #            QMessageBox.critical(self, "Error", f"An error occurred during heatmap generation: {e}")
    #    else:
    #        QMessageBox.warning(self, "No Data", "Please load a CSV file first.")
#
    #def plot_map(self):       
    #    """Launches the Streamlit application."""
    #    # Check if Streamlit is already running
    #    if self.streamlit_process is not None:
    #        QMessageBox.information(self, "Streamlit Already Running", "Streamlit application is already running.")
    #        return         
       
    

if __name__ in "__main__":
    app = QApplication([sys.argv])
    main_window = MagnavisApp()
    main_window.show()
    app.exec_()