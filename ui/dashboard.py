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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_quality import (
    DataQualityChecker,
    FlightDataValidator,
    DataQualityReport,
    BookingDataQualityReport,
    BookingCompletionAnalyzer,
    BookingOutlierDetector,
    BookingConsistencyValidator,
    BookingDuplicateAndOriginAnalyzer,
    detect_dataset_type,
    PSEUDO_MISSING_VALUE,
)

# Centralized color definitions for severity and issue types
SEVERITY_COLORS = {
    "High": "#EF553B",
    "Medium": "#FECB52",
    "Low": "#636EFA",
}

SEVERITY_BG_COLORS = {
    "High": "background-color: #ffcccc",
    "Medium": "background-color: #fff3cd",
    "Low": "background-color: #cce5ff",
}

ISSUE_TYPE_COLORS = {
    "Same Airport": SEVERITY_COLORS["High"],
    "Distance Mismatch": SEVERITY_COLORS["Medium"],
}

# Booking completion rate baseline reference (from initial data analysis)
COMPLETION_RATE_BASELINE = 14.96

# Threshold below which a high-volume origin is flagged as an optimization opportunity
OPPORTUNITY_COMPLETION_THRESHOLD = 10


# Page configuration
st.set_page_config(
    page_title="Flight Data Quality Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


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
            help="Total number of flight records",
        )

    with col2:
        st.metric(
            "Total Columns",
            report["overview"]["total_columns"],
            help="Number of data attributes",
        )

    with col3:
        memory_mb = report["overview"]["memory_usage"] / (1024 * 1024)
        st.metric(
            "Memory Usage", f"{memory_mb:.2f} MB", help="Dataset memory footprint"
        )

    with col4:
        duplicate_pct = report["duplicates"]["duplicate_percentage"]
        st.metric(
            "Duplicate %", f"{duplicate_pct}%", help="Percentage of duplicate records"
        )


def display_null_analysis(report: dict):
    """Display null value analysis"""
    st.header("🔍 Missing Data Analysis")

    null_data = report["null_analysis"]
    null_df = pd.DataFrame(list(null_data.items()), columns=["Column", "Null %"])
    null_df = null_df.sort_values("Null %", ascending=False)

    # Create visualization
    fig = px.bar(
        null_df,
        x="Column",
        y="Null %",
        title="Missing Data by Column",
        color="Null %",
        color_continuous_scale="Reds",
        labels={"Null %": "Missing Data %"},
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Show table
    st.subheader("Detailed Null Statistics")
    st.dataframe(
        null_df.style.background_gradient(cmap="Reds", subset=["Null %"]),
        use_container_width=True,
    )


def display_statistics(report: dict):
    """Display statistical analysis"""
    st.header("📈 Statistical Summary")

    stats_data = report["statistics"]

    if not stats_data:
        st.warning("No numeric columns found for statistical analysis")
        return

    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Distribution Analysis", "Statistical Table"])

    with tab1:
        # Select column to visualize
        available_cols = list(stats_data.keys())
        selected_col = st.selectbox("Select column to analyze", available_cols)

        if selected_col and "error" not in stats_data[selected_col]:
            col_stats = stats_data[selected_col]

            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Count", f"{col_stats['count']:,}")
            with col2:
                st.metric(
                    "Mean", f"{col_stats['mean']:.2f}" if col_stats["mean"] else "N/A"
                )
            with col3:
                st.metric(
                    "Median",
                    f"{col_stats['median']:.2f}" if col_stats["median"] else "N/A",
                )
            with col4:
                st.metric(
                    "Min", f"{col_stats['min']:.2f}" if col_stats["min"] else "N/A"
                )
            with col5:
                st.metric(
                    "Max", f"{col_stats['max']:.2f}" if col_stats["max"] else "N/A"
                )

    with tab2:
        # Create statistics table
        stats_df = pd.DataFrame(stats_data).T
        st.dataframe(stats_df, use_container_width=True)


def display_validations(report: dict, df: pd.DataFrame):
    """Display validation results"""
    st.header("✅ Data Validation Results")

    validations = report["validations"]

    # Create metrics for each validation
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Invalid Carrier Codes",
            validations["invalid_carrier_codes"],
            delta=None if validations["invalid_carrier_codes"] == 0 else "Issues found",
            delta_color="inverse",
        )
        st.metric(
            "Invalid Airport Codes",
            validations["invalid_airport_codes"],
            delta=None if validations["invalid_airport_codes"] == 0 else "Issues found",
            delta_color="inverse",
        )

    with col2:
        st.metric(
            "Invalid Dates",
            validations["invalid_dates"],
            delta=None if validations["invalid_dates"] == 0 else "Issues found",
            delta_color="inverse",
        )
        st.metric(
            "Invalid Delay Logic",
            validations["invalid_delay_logic"],
            delta=None if validations["invalid_delay_logic"] == 0 else "Issues found",
            delta_color="inverse",
        )

    with col3:
        st.metric(
            "Invalid Distances",
            validations["invalid_distances"],
            delta=None if validations["invalid_distances"] == 0 else "Issues found",
            delta_color="inverse",
        )
        st.metric(
            "Same Origin-Destination",
            validations.get("same_origin_destination", 0),
            delta=None
            if validations.get("same_origin_destination", 0) == 0
            else "Issues found",
            delta_color="inverse",
        )
        st.metric(
            "Distance Mismatches",
            validations.get("distance_mismatches", 0),
            delta=None
            if validations.get("distance_mismatches", 0) == 0
            else "Issues found",
            delta_color="inverse",
        )

    # Validation summary chart
    validation_df = pd.DataFrame(
        list(validations.items()), columns=["Validation", "Issues"]
    )
    fig = px.bar(
        validation_df,
        x="Validation",
        y="Issues",
        title="Data Validation Issues Summary",
        color="Issues",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)


def display_route_validation(report: dict, df: pd.DataFrame):
    """Display route validation results section"""
    st.header("🛤️ Route Validation Results")

    validations = report["validations"]
    route_details = report.get("route_validation_details", [])

    # Summary cards
    col1, col2, col3 = st.columns(3)

    same_airport_count = validations.get("same_origin_destination", 0)
    distance_mismatch_count = validations.get("distance_mismatches", 0)
    total_flagged = same_airport_count + distance_mismatch_count

    with col1:
        st.metric(
            "Same Airport Issues",
            same_airport_count,
            delta="Issues found" if same_airport_count > 0 else None,
            delta_color="inverse",
        )

    with col2:
        st.metric(
            "Distance Mismatches",
            distance_mismatch_count,
            delta="Issues found" if distance_mismatch_count > 0 else None,
            delta_color="inverse",
        )

    with col3:
        st.metric(
            "Total Flagged Routes",
            total_flagged,
            delta="Issues found" if total_flagged > 0 else None,
            delta_color="inverse",
        )

    if not route_details:
        st.success("No route validation issues found!")
        return

    details_df = pd.DataFrame(route_details)

    # Visualizations
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        # Bar chart: Issues by type
        type_counts = details_df["issue_type"].value_counts().reset_index()
        type_counts.columns = ["Issue Type", "Count"]
        fig_bar = px.bar(
            type_counts,
            x="Issue Type",
            y="Count",
            title="Issues by Type",
            color="Issue Type",
            color_discrete_map=ISSUE_TYPE_COLORS,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with viz_col2:
        # Pie chart: Severity distribution
        severity_counts = details_df["severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        fig_pie = px.pie(
            severity_counts,
            values="Count",
            names="Severity",
            title="Severity Distribution",
            color="Severity",
            color_discrete_map=SEVERITY_COLORS,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Filter by issue type
    st.subheader("Flagged Routes")
    issue_types = ["All"] + sorted(details_df["issue_type"].unique().tolist())
    selected_type = st.selectbox("Filter by validation type", issue_types)

    filtered_df = details_df
    if selected_type != "All":
        filtered_df = details_df[details_df["issue_type"] == selected_type]

    # Color-code severity in the table
    def severity_color(val):
        return SEVERITY_BG_COLORS.get(val, "")

    st.dataframe(
        filtered_df.style.map(severity_color, subset=["severity"]),
        use_container_width=True,
    )

    # CSV export
    csv_data = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Export Flagged Routes to CSV",
        data=csv_data,
        file_name="flagged_routes.csv",
        mime="text/csv",
    )


def display_flight_insights(df: pd.DataFrame):
    """Display flight-specific insights"""
    st.header("✈️ Flight Insights")

    col1, col2 = st.columns(2)

    with col1:
        # Top carriers by flight count
        if "op_unique_carrier" in df.columns:
            st.subheader("Top Carriers")
            carrier_counts = df["op_unique_carrier"].value_counts().head(10)
            fig = px.bar(
                x=carrier_counts.index,
                y=carrier_counts.values,
                labels={"x": "Carrier", "y": "Number of Flights"},
                title="Top 10 Carriers by Flight Count",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cancellation rate
        if "cancelled" in df.columns:
            st.subheader("Cancellation Analysis")
            cancellation_rate = (df["cancelled"].sum() / len(df)) * 100

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=cancellation_rate,
                    title={"text": "Cancellation Rate (%)"},
                    gauge={
                        "axis": {"range": [None, 10]},
                        "bar": {"color": "darkred"},
                        "steps": [
                            {"range": [0, 2], "color": "lightgreen"},
                            {"range": [2, 5], "color": "yellow"},
                            {"range": [5, 10], "color": "red"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    # Delay analysis
    if "dep_delay" in df.columns and "arr_delay" in df.columns:
        st.subheader("Delay Distribution")

        delay_col = st.radio("Select delay type", ["dep_delay", "arr_delay"])

        delay_data = df[delay_col].dropna()
        fig = px.histogram(
            delay_data,
            x=delay_col,
            nbins=50,
            title=f'{delay_col.replace("_", " ").title()} Distribution',
            labels={delay_col: "Delay (minutes)"},
        )
        st.plotly_chart(fig, use_container_width=True)


def display_booking_overview(df: pd.DataFrame, report: dict):
    """Display overview section for booking data"""
    st.header("📊 Booking Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Bookings",
            f"{report['overview']['total_rows']:,}",
            help="Total number of booking records",
        )

    with col2:
        st.metric(
            "Completion Rate",
            f"{report['overview']['completion_rate']}%",
            help="Percentage of bookings completed",
        )

    with col3:
        st.metric(
            "Duplicates",
            f"{report['overview']['duplicate_count']:,}",
            help="Number of duplicate records",
        )

    with col4:
        st.metric(
            "Total Columns",
            report["overview"]["total_columns"],
            help="Number of data attributes",
        )


def display_booking_null_note(report: dict):
    """Display additional note about (not set) values in booking_origin"""
    not_set_count = report.get("not_set_booking_origin", 0)
    if not_set_count > 0:
        st.subheader("Pseudo-Missing Data")
        st.warning(
            f"**booking_origin** contains {not_set_count:,} records with value "
            f'"{PSEUDO_MISSING_VALUE}", which may represent missing data.'
        )


def display_completion_analytics(df: pd.DataFrame):
    """Display booking completion rate analytics dashboard"""
    st.header("📈 Completion Rate Analytics")

    analyzer = BookingCompletionAnalyzer(df)

    # Overall completion rate gauge
    overall = analyzer.overall_completion_rate()
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall["completion_rate"],
            number={"suffix": "%"},
            title={"text": "Overall Completion Rate"},
            delta={"reference": COMPLETION_RATE_BASELINE, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#636EFA"},
                "steps": [
                    {"range": [0, 10], "color": "#ffcccc"},
                    {"range": [10, 20], "color": "#fff3cd"},
                    {"range": [20, 50], "color": "#cce5ff"},
                    {"range": [50, 100], "color": "#ccffcc"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": COMPLETION_RATE_BASELINE,
                },
            },
        )
    )
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(
        f"**{overall['completed']:,}** of **{overall['total_bookings']:,}** "
        f"bookings completed ({overall['completion_rate']}%)"
    )

    st.markdown("---")

    # Channel and Trip Type side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Sales Channel")
        channel_data = analyzer.completion_by_channel()
        fig_channel = px.bar(
            channel_data,
            x="sales_channel",
            y="completion_rate",
            text="completion_rate",
            title="Completion Rate by Sales Channel",
            labels={
                "sales_channel": "Sales Channel",
                "completion_rate": "Completion Rate (%)",
            },
            color="sales_channel",
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        fig_channel.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_channel, use_container_width=True)

    with col2:
        st.subheader("By Trip Type")
        trip_data = analyzer.completion_by_trip_type()
        fig_trip = px.bar(
            trip_data,
            x="trip_type",
            y="completion_rate",
            text="completion_rate",
            title="Completion Rate by Trip Type",
            labels={
                "trip_type": "Trip Type",
                "completion_rate": "Completion Rate (%)",
            },
            color="trip_type",
            color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
        )
        fig_trip.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_trip, use_container_width=True)

    st.markdown("---")

    # Booking Origin (top 15) horizontal bar chart
    st.subheader("By Booking Origin (Top 15)")
    origin_data = analyzer.completion_by_origin(top_n=15)
    fig_origin = px.bar(
        origin_data,
        y="booking_origin",
        x="completion_rate",
        text="completion_rate",
        orientation="h",
        title="Completion Rate by Booking Origin (Top 15 by Volume)",
        labels={
            "booking_origin": "Booking Origin",
            "completion_rate": "Completion Rate (%)",
        },
        color="completion_rate",
        color_continuous_scale="RdYlGn",
    )
    fig_origin.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_origin.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_origin, use_container_width=True)

    st.markdown("---")

    # Extras and Flight Day side by side
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("By Number of Extras (0-3)")
        extras_data = analyzer.completion_by_extras()
        fig_extras = px.line(
            extras_data,
            x="num_extras",
            y="completion_rate",
            markers=True,
            title="Completion Rate by Extras Requested",
            labels={
                "num_extras": "Number of Extras",
                "completion_rate": "Completion Rate (%)",
            },
        )
        fig_extras.update_traces(
            line=dict(color="#636EFA", width=3),
            marker=dict(size=10),
        )
        st.plotly_chart(fig_extras, use_container_width=True)

    with col4:
        st.subheader("By Flight Day")
        day_data = analyzer.completion_by_flight_day()
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_data["flight_day"] = pd.Categorical(
            day_data["flight_day"], categories=day_order, ordered=True
        )
        day_data = day_data.sort_values("flight_day")
        fig_day = px.imshow(
            [day_data["completion_rate"].values],
            x=day_data["flight_day"].values,
            y=["Completion Rate"],
            color_continuous_scale="RdYlGn",
            title="Completion Rate Heatmap by Flight Day",
            text_auto=".1f",
            aspect="auto",
        )
        fig_day.update_layout(height=200)
        st.plotly_chart(fig_day, use_container_width=True)

    st.markdown("---")

    # Key Insight: Volume vs Completion
    st.subheader("Key Insight: Volume vs Completion")
    vol_data = analyzer.volume_vs_completion()
    vol_data = vol_data.sort_values("total", ascending=False).head(10)

    # Highlight the biggest opportunity
    if len(vol_data) > 0:
        max_vol = vol_data.iloc[0]
        if max_vol["completion_rate"] < OPPORTUNITY_COMPLETION_THRESHOLD:
            st.warning(
                f"**{max_vol['booking_origin']}** accounts for "
                f"**{max_vol['volume_pct']}%** of booking volume but has only "
                f"**{max_vol['completion_rate']}%** completion rate "
                f"— biggest optimization opportunity!"
            )

    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Bar(
            x=vol_data["booking_origin"],
            y=vol_data["volume_pct"],
            name="Volume %",
            marker_color="#636EFA",
        )
    )
    fig_vol.add_trace(
        go.Bar(
            x=vol_data["booking_origin"],
            y=vol_data["completion_rate"],
            name="Completion Rate %",
            marker_color="#EF553B",
        )
    )
    fig_vol.update_layout(
        title="Volume Share vs Completion Rate by Origin",
        barmode="group",
        xaxis_title="Booking Origin",
        yaxis_title="Percentage (%)",
    )
    st.plotly_chart(fig_vol, use_container_width=True)


def display_booking_outliers(df: pd.DataFrame):
    """Display outlier detection for booking numeric fields"""
    st.header("🔎 Outlier Detection")

    # Sidebar configurable thresholds
    with st.sidebar:
        st.subheader("Outlier Thresholds")
        max_passengers = st.number_input(
            "Max passengers", min_value=1, max_value=50, value=6
        )
        max_purchase_lead = st.number_input(
            "Max purchase lead (days)", min_value=1, max_value=1000, value=365
        )
        max_length_of_stay = st.number_input(
            "Max length of stay (days)", min_value=1, max_value=1000, value=365
        )

    detector = BookingOutlierDetector(df)

    # Threshold-based flagging results
    st.subheader("Threshold-Based Flags")

    thresholds = {
        "num_passengers": ("Passengers", max_passengers),
        "purchase_lead": ("Purchase Lead (days)", max_purchase_lead),
        "length_of_stay": ("Length of Stay (days)", max_length_of_stay),
    }

    cols = st.columns(len(thresholds))
    for col_ui, (col_name, (label, max_val)) in zip(cols, thresholds.items()):
        flagged = detector.flag_by_threshold(col_name, max_value=max_val)
        with col_ui:
            st.metric(
                f"{label} > {max_val}",
                f"{len(flagged):,}",
                delta="Issues found" if len(flagged) > 0 else None,
                delta_color="inverse",
            )

    st.markdown("---")

    # IQR outlier summary table
    st.subheader("IQR-Based Outlier Summary")
    numeric_cols = [
        "num_passengers",
        "purchase_lead",
        "length_of_stay",
        "flight_duration",
    ]
    available_cols = [c for c in numeric_cols if c in df.columns]
    summary = detector.iqr_outlier_summary(available_cols)

    if not summary.empty:
        st.dataframe(
            summary.style.background_gradient(
                cmap="Reds", subset=["outlier_count", "outlier_pct"]
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # Box plots for each numeric column
    st.subheader("Distribution Box Plots")
    for col_name in available_cols:
        fig = px.box(
            df,
            y=col_name,
            title=f"Distribution of {col_name}",
            labels={col_name: col_name.replace("_", " ").title()},
            points="outliers",
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def display_booking_consistency(df: pd.DataFrame):
    """Display logical consistency validation for bookings"""
    st.header("🔗 Logical Consistency Checks")

    validator = BookingConsistencyValidator(df)
    summary = validator.inconsistency_summary()

    # Summary table with severity coloring
    st.subheader("Inconsistency Summary")

    def severity_style(val):
        colors = {
            "High": "background-color: #ffcccc; color: #de350b; font-weight: bold",
            "Medium": "background-color: #fff3cd; color: #ff8b00; font-weight: bold",
            "Info": "background-color: #deebff; color: #0052cc; font-weight: bold",
        }
        return colors.get(val, "")

    st.dataframe(
        summary.style.map(severity_style, subset=["severity"]),
        use_container_width=True,
    )

    st.markdown("---")

    # Metric cards
    st.subheader("Flagged Records")
    col1, col2, col3, col4 = st.columns(4)

    roundtrip_zero = validator.flag_roundtrip_zero_stay()
    oneway_positive = validator.flag_oneway_positive_stay()
    same_day = validator.flag_same_day_bookings()

    with col1:
        st.metric(
            "RoundTrip + Zero Stay",
            f"{len(roundtrip_zero):,}",
            delta="Medium" if len(roundtrip_zero) > 0 else None,
            delta_color="off",
        )
    with col2:
        st.metric(
            "OneWay + Positive Stay",
            f"{len(oneway_positive):,}",
            delta="High" if len(oneway_positive) > 0 else None,
            delta_color="inverse",
        )
    with col3:
        circle = (
            df[df["trip_type"] == "CircleTrip"]
            if "trip_type" in df.columns
            else pd.DataFrame()
        )
        st.metric(
            "CircleTrip Records",
            f"{len(circle):,}",
            delta="Medium" if len(circle) > 0 else None,
            delta_color="off",
        )
    with col4:
        st.metric(
            "Same-Day Bookings",
            f"{len(same_day):,}",
            delta="Info" if len(same_day) > 0 else None,
            delta_color="off",
        )

    st.markdown("---")

    # Drill-down tables
    if len(roundtrip_zero) > 0:
        with st.expander(f"RoundTrip with Zero Stay ({len(roundtrip_zero)} records)"):
            st.dataframe(roundtrip_zero.head(50), use_container_width=True)
            if len(roundtrip_zero) > 50:
                st.info(f"Showing first 50 of {len(roundtrip_zero)} records")

    if len(oneway_positive) > 0:
        with st.expander(f"OneWay with Positive Stay ({len(oneway_positive)} records)"):
            st.dataframe(oneway_positive.head(50), use_container_width=True)
            if len(oneway_positive) > 50:
                st.info(f"Showing first 50 of {len(oneway_positive)} records")

    if len(same_day) > 0:
        with st.expander(f"Same-Day Bookings ({len(same_day)} records)"):
            st.dataframe(same_day.head(50), use_container_width=True)
            if len(same_day) > 50:
                st.info(f"Showing first 50 of {len(same_day)} records")


@st.cache_data
def _convert_to_csv(df: pd.DataFrame) -> str:
    """Cache CSV conversion to avoid recomputing on every rerun."""
    return df.to_csv(index=False)


def display_duplicate_and_origin_analysis(df: pd.DataFrame):
    """Display duplicate detection and booking origin cleanup analysis"""
    st.header("🔁 Duplicate Detection & Origin Cleanup")

    analyzer = BookingDuplicateAndOriginAnalyzer(df)

    # Compute all results once to avoid redundant passes
    preview = analyzer.dedup_preview()
    not_set = analyzer.flag_not_set_origins()
    low_vol = analyzer.low_volume_origins()
    summary = analyzer.origin_quality_summary(not_set=not_set, low_vol=low_vol)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dup_pct = (
            round((preview["duplicates_count"] / len(df)) * 100, 2)
            if len(df) > 0
            else 0
        )
        st.metric(
            "Exact Duplicates",
            f"{preview['duplicates_count']:,}",
            delta=f"{dup_pct}%" if preview["duplicates_count"] > 0 else None,
            delta_color="inverse",
        )
    with col2:
        st.metric(
            "Clean Row Count",
            f"{preview['clean_count']:,}",
        )
    with col3:
        st.metric(
            '"(not set)" Origins',
            f"{len(not_set):,}",
            delta="Issues found" if len(not_set) > 0 else None,
            delta_color="inverse",
        )
    with col4:
        st.metric(
            "Low-Volume Origins (<5)",
            f"{len(low_vol):,}",
            delta=f"of {summary['total_origins']} total" if len(low_vol) > 0 else None,
            delta_color="off",
        )

    st.markdown("---")

    # Duplicate preview
    st.subheader("Duplicate Rows Preview")
    if preview["duplicates_count"] > 0:
        with st.expander(f"View {preview['duplicates_count']:,} duplicate rows"):
            st.dataframe(preview["duplicate_rows"].head(50), use_container_width=True)
            if preview["duplicates_count"] > 50:
                st.info(
                    f"Showing first 50 of {preview['duplicates_count']:,} duplicates"
                )
    else:
        st.success("No duplicate rows found!")

    st.markdown("---")

    # Low-volume origins table
    st.subheader("Low-Volume Booking Origins")
    if len(low_vol) > 0:
        fig = px.bar(
            low_vol.sort_values("count"),
            x="count",
            y="booking_origin",
            orientation="h",
            title=f"Origins with Fewer Than 5 Bookings ({len(low_vol)} countries)",
            labels={"count": "Bookings", "booking_origin": "Origin"},
            color="count",
            color_continuous_scale="Reds_r",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(300, len(low_vol) * 25),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("All origins have sufficient booking volume.")

    st.markdown("---")

    # Export cleaned dataset
    st.subheader("Export Cleaned Dataset")
    clean = analyzer.clean_dataset()
    csv_data = _convert_to_csv(clean)
    st.download_button(
        label=f"📥 Download De-duplicated Dataset ({len(clean):,} rows)",
        data=csv_data,
        file_name="customer_booking_cleaned.csv",
        mime="text/csv",
    )


def display_download_section(report: dict, report_filename: str):
    """Display download buttons for report exports"""
    import json

    st.markdown("---")
    st.header("📥 Download Reports")

    col1, col2 = st.columns(2)

    with col1:
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=report_filename,
            mime="application/json",
        )

    with col2:
        null_df = pd.DataFrame(
            list(report["null_analysis"].items()),
            columns=["Column", "Null_Percentage"],
        )
        csv = null_df.to_csv(index=False)
        st.download_button(
            label="Download Null Analysis CSV",
            data=csv,
            file_name="null_analysis.csv",
            mime="text/csv",
        )


def main():
    """Main dashboard application"""
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # File upload or use default
        uploaded_file = st.file_uploader(
            "Upload CSV data",
            type=["csv"],
            help="Upload a flight or customer booking CSV file",
        )

        if uploaded_file is None:
            # Use default sample data
            default_path = (
                Path(__file__).parent.parent / "data" / "flight_data_2024_sample.csv"
            )
            if default_path.exists():
                st.info(f"Using sample data: {default_path.name}")
                file_path = str(default_path)
            else:
                st.error("No data file found. Please upload a CSV file.")
                return
        else:
            # Save uploaded file temporarily
            temp_path = Path(__file__).parent.parent / "data" / uploaded_file.name
            with open(temp_path, "wb") as f:
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
            dataset_type = detect_dataset_type(df)

        if dataset_type == "booking":
            st.title("📋 Customer Booking Data Quality Dashboard")

            with st.spinner("Generating booking quality report..."):
                report_gen = BookingDataQualityReport(df)
                report = report_gen.generate()

            display_booking_overview(df, report)
            st.markdown("---")

            display_completion_analytics(df)
            st.markdown("---")

            display_booking_outliers(df)
            st.markdown("---")

            display_booking_consistency(df)
            st.markdown("---")

            display_duplicate_and_origin_analysis(df)
            st.markdown("---")

            display_null_analysis(report)
            display_booking_null_note(report)
            st.markdown("---")

            display_statistics(report)

            display_download_section(report, "booking_quality_report.json")

        else:
            st.title("✈️ Flight Data Quality Dashboard")

            with st.spinner("Generating flight quality report..."):
                report_gen = DataQualityReport(df)
                report = report_gen.generate()

            # Display sections
            display_overview(df, report)
            st.markdown("---")

            display_null_analysis(report)
            st.markdown("---")

            display_validations(report, df)
            st.markdown("---")

            display_route_validation(report, df)
            st.markdown("---")

            display_statistics(report)
            st.markdown("---")

            display_flight_insights(df)

            display_download_section(report, "data_quality_report.json")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
