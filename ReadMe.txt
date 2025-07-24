MagNavis: Magnetic Navigation Data Analysis & Visualization Tool

---

Overview
--------
MagNavis is a powerful, interactive desktop application for the analysis and visualization of magnetic navigation data. Built with PyQt5, it provides a modern GUI for importing, processing, analyzing, and visualizing geospatial and time-series magnetic data. The tool is especially focused on anomaly detection, signal processing, and map-based visualization.

Key Features
------------
- Data Import & Management:
  - Import CSV/Excel files.
  - Preview and manage multiple datasets.
  - Flexible column mapping for latitude, longitude, magnetic values, etc.
- Data Insights:
  - Data statistics and area coverage.
  - CDF (Cumulative Distribution Function) analysis.
  - Frequency spectrum and power spectral density.
  - Time series visualization.
- Anomaly Detection:
  - Multiple methods: Threshold, Quantile, IQR, ESD, Persistence, Level Shift, Volatility, Seasonal, Autoregression.
  - Interactive parameter selection and result visualization.
- Data Processing Tools:
  - Low/High/Band Pass Filters.
  - Time lag correction.
  - Residual magnetic anomaly calculation.
  - Analytic signal generation.
  - Outlier removal, rolling statistics, normalization/standardization.
- Visualization:
  - 2D/3D plots, heatmaps, cluster maps, PCA, and more.
  - Map-based visualizations using Folium and Leafmap.
  - Interactive plots with hover tooltips.

Installation
------------
1. Clone or Download the Repository
2. (Recommended) Create a Virtual Environment
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
3. Install Dependencies
   pip install -r Requirement.txt

   Dependencies include:
   requests, folium, leafmap, PyQt5, PyQtWebEngine, pandas, numpy, seaborn, matplotlib, plotly, scipy, scikit-learn, contextily, geopandas, cartopy, scikit-image, adtk

Usage
-----
1. Run the Application
   python MagNavis_v3.py
2. Workflow
   - Import your data file (CSV/Excel).
   - Map columns to standard names (Latitude, Longitude, Magnetic_Values, etc.).
   - Use the sidebar to select data insights, anomaly detection methods, processing tools, and visualization options.
   - Adjust parameters as needed and view results in real time.
3. Visualization
   - Explore your data with interactive plots and maps.
   - Export processed data or analytic signals as CSV.

File Structure
--------------
- MagNavis_v3.py — Main application (recommended entry point)
- Requirement.txt — List of required Python packages
- Code Overview.rtf — Detailed code and feature overview
- interactive_map.html — (Large) HTML output for map visualization (optional)
- main.py, MagNavis_v2.py — Previous versions (for reference)
- venv/ — (Optional) Virtual environment directory

Notes
-----
- The application is designed for desktop use and requires a Python 3 environment.
- For best results, use the latest version of all dependencies.
- The interactive_map.html file is a large, generated output and not required for running the application.

Credits
-------
Developed by:
- TG MagNav