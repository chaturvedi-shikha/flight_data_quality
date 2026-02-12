"""
Streamlit Dashboard for Flight Data Quality Reports
Interactive UI for exploring data quality metrics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_quality import DataQualityChecker, FlightDataValidator, DataQualityReport


# Page configuration
st.set_page_config(
    page_title="Flight Data Quality Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load flight data from CSV"""
    return pd.read_csv(file_path)


def display_overview(df: pd.DataFrame, report: dict):
    """Display overview section"""
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Flights",
            f"{report['overview']['total_rows']:,}",
            help="Total number of flight records"
        )
    
    with col2:
        st.metric(
            "Total Columns",
            report['overview']['total_columns'],
            help="Number of data attributes"
        )
    
    with col3:
        memory_mb = report['overview']['memory_usage'] / (1024 * 1024)
        st.metric(
            "Memory Usage",
            f"{memory_mb:.2f} MB",
            help="Dataset memory footprint"
        )
    
    with col4:
        duplicate_pct = report['duplicates']['duplicate_percentage']
        st.metric(
            "Duplicate %",
            f"{duplicate_pct}%",
            help="Percentage of duplicate records"
        )


def display_null_analysis(report: dict):
    """Display null value analysis"""
    st.header("🔍 Missing Data Analysis")
    
    null_data = report['null_analysis']
    null_df = pd.DataFrame(list(null_data.items()), columns=['Column', 'Null %'])
    null_df = null_df.sort_values('Null %', ascending=False)
    
    # Create visualization
    fig = px.bar(
        null_df,
        x='Column',
        y='Null %',
        title='Missing Data by Column',
        color='Null %',
        color_continuous_scale='Reds',
        labels={'Null %': 'Missing Data %'}
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table
    st.subheader("Detailed Null Statistics")
    st.dataframe(
        null_df.style.background_gradient(cmap='Reds', subset=['Null %']),
        use_container_width=True
    )


def display_statistics(report: dict):
    """Display statistical analysis"""
    st.header("📈 Statistical Summary")
    
    stats_data = report['statistics']
    
    if not stats_data:
        st.warning("No numeric columns found for statistical analysis")
        return
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Distribution Analysis", "Statistical Table"])
    
    with tab1:
        # Select column to visualize
        available_cols = list(stats_data.keys())
        selected_col = st.selectbox("Select column to analyze", available_cols)
        
        if selected_col and 'error' not in stats_data[selected_col]:
            col_stats = stats_data[selected_col]
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Count", f"{col_stats['count']:,}")
            with col2:
                st.metric("Mean", f"{col_stats['mean']:.2f}" if col_stats['mean'] else "N/A")
            with col3:
                st.metric("Median", f"{col_stats['median']:.2f}" if col_stats['median'] else "N/A")
            with col4:
                st.metric("Min", f"{col_stats['min']:.2f}" if col_stats['min'] else "N/A")
            with col5:
                st.metric("Max", f"{col_stats['max']:.2f}" if col_stats['max'] else "N/A")
    
    with tab2:
        # Create statistics table
        stats_df = pd.DataFrame(stats_data).T
        st.dataframe(stats_df, use_container_width=True)


def display_validations(report: dict, df: pd.DataFrame):
    """Display validation results"""
    st.header("✅ Data Validation Results")
    
    validations = report['validations']
    
    # Create metrics for each validation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Invalid Carrier Codes",
            validations['invalid_carrier_codes'],
            delta=None if validations['invalid_carrier_codes'] == 0 else "Issues found",
            delta_color="inverse"
        )
        st.metric(
            "Invalid Airport Codes",
            validations['invalid_airport_codes'],
            delta=None if validations['invalid_airport_codes'] == 0 else "Issues found",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Invalid Dates",
            validations['invalid_dates'],
            delta=None if validations['invalid_dates'] == 0 else "Issues found",
            delta_color="inverse"
        )
        st.metric(
            "Invalid Delay Logic",
            validations['invalid_delay_logic'],
            delta=None if validations['invalid_delay_logic'] == 0 else "Issues found",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Invalid Distances",
            validations['invalid_distances'],
            delta=None if validations['invalid_distances'] == 0 else "Issues found",
            delta_color="inverse"
        )
        st.metric(
            "Same Origin-Destination",
            validations.get('same_origin_destination', 0),
            delta=None if validations.get('same_origin_destination', 0) == 0 else "Issues found",
            delta_color="inverse"
        )
    
    # Validation summary chart
    validation_df = pd.DataFrame(list(validations.items()), columns=['Validation', 'Issues'])
    fig = px.bar(
        validation_df,
        x='Validation',
        y='Issues',
        title='Data Validation Issues Summary',
        color='Issues',
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def display_flight_insights(df: pd.DataFrame):
    """Display flight-specific insights"""
    st.header("✈️ Flight Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top carriers by flight count
        if 'op_unique_carrier' in df.columns:
            st.subheader("Top Carriers")
            carrier_counts = df['op_unique_carrier'].value_counts().head(10)
            fig = px.bar(
                x=carrier_counts.index,
                y=carrier_counts.values,
                labels={'x': 'Carrier', 'y': 'Number of Flights'},
                title='Top 10 Carriers by Flight Count'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cancellation rate
        if 'cancelled' in df.columns:
            st.subheader("Cancellation Analysis")
            cancellation_rate = (df['cancelled'].sum() / len(df)) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cancellation_rate,
                title={'text': "Cancellation Rate (%)"},
                gauge={'axis': {'range': [None, 10]},
                       'bar': {'color': "darkred"},
                       'steps': [
                           {'range': [0, 2], 'color': "lightgreen"},
                           {'range': [2, 5], 'color': "yellow"},
                           {'range': [5, 10], 'color': "red"}
                       ]}
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    # Delay analysis
    if 'dep_delay' in df.columns and 'arr_delay' in df.columns:
        st.subheader("Delay Distribution")
        
        delay_col = st.radio("Select delay type", ['dep_delay', 'arr_delay'])
        
        delay_data = df[delay_col].dropna()
        fig = px.histogram(
            delay_data,
            x=delay_col,
            nbins=50,
            title=f'{delay_col.replace("_", " ").title()} Distribution',
            labels={delay_col: 'Delay (minutes)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application"""
    st.title("✈️ Flight Data Quality Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File upload or use default
        uploaded_file = st.file_uploader(
            "Upload flight data CSV",
            type=['csv'],
            help="Upload your flight data CSV file"
        )
        
        if uploaded_file is None:
            # Use default sample data
            default_path = Path(__file__).parent.parent / 'data' / 'flight_data_2024_sample.csv'
            if default_path.exists():
                st.info(f"Using sample data: {default_path.name}")
                file_path = str(default_path)
            else:
                st.error("No data file found. Please upload a CSV file.")
                return
        else:
            # Save uploaded file temporarily
            temp_path = Path(__file__).parent.parent / 'data' / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_path = str(temp_path)
            st.success(f"Loaded: {uploaded_file.name}")
        
        # Generate report button
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.rerun()
    
    # Load data and generate report
    try:
        with st.spinner("Loading data and generating quality report..."):
            df = load_data(file_path)
            
            # Generate report
            report_gen = DataQualityReport(df)
            report = report_gen.generate()
        
        # Display sections
        display_overview(df, report)
        st.markdown("---")
        
        display_null_analysis(report)
        st.markdown("---")
        
        display_validations(report, df)
        st.markdown("---")
        
        display_statistics(report)
        st.markdown("---")
        
        display_flight_insights(df)
        
        # Download section
        st.markdown("---")
        st.header("📥 Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            import json
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name="data_quality_report.json",
                mime="application/json"
            )
        
        with col2:
            # CSV download (null analysis)
            null_df = pd.DataFrame(list(report['null_analysis'].items()), 
                                   columns=['Column', 'Null_Percentage'])
            csv = null_df.to_csv(index=False)
            st.download_button(
                label="Download Null Analysis CSV",
                data=csv,
                file_name="null_analysis.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
