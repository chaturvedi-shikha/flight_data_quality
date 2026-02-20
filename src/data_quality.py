"""
Data Quality Analysis Module for Flight Data
Implements comprehensive data quality checks and reporting
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from geopy.distance import geodesic


class DataQualityChecker:
    """Core class for data quality checks"""

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize DataQualityChecker

        Args:
            filepath: Path to CSV file to analyze
        """
        self.df = None
        if filepath:
            self.load_data(filepath)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        self.df = pd.read_csv(filepath)
        return self.df

    def calculate_null_percentage(self) -> Dict[str, float]:
        """
        Calculate percentage of null values for each column

        Returns:
            Dictionary mapping column names to null percentages
        """
        if self.df is None:
            raise ValueError("No data loaded")

        null_pct = {}
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            total_count = len(self.df)
            null_pct[col] = round((null_count / total_count) * 100, 2)

        return null_pct

    def check_duplicates(self, subset: Optional[List[str]] = None) -> int:
        """
        Check for duplicate rows

        Args:
            subset: List of columns to check for duplicates. If None, check all columns

        Returns:
            Number of duplicate rows
        """
        if self.df is None:
            raise ValueError("No data loaded")

        duplicates = self.df.duplicated(subset=subset)
        return duplicates.sum()

    def calculate_statistics(self, column: str) -> Dict[str, float]:
        """
        Calculate statistical summary for a numeric column

        Args:
            column: Column name to analyze

        Returns:
            Dictionary of statistics (mean, median, std, min, max, count)
        """
        if self.df is None:
            raise ValueError("No data loaded")

        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")

        col_data = self.df[column].dropna()

        return {
            "count": int(len(col_data)),
            "mean": float(col_data.mean()) if len(col_data) > 0 else None,
            "median": float(col_data.median()) if len(col_data) > 0 else None,
            "std": float(col_data.std()) if len(col_data) > 0 else None,
            "min": float(col_data.min()) if len(col_data) > 0 else None,
            "max": float(col_data.max()) if len(col_data) > 0 else None,
        }

    def detect_outliers(
        self, column: str, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using IQR method

        Args:
            column: Column name to analyze
            method: Detection method ('iqr' or 'zscore')
            threshold: IQR multiplier (default 1.5 for standard outlier detection)

        Returns:
            DataFrame containing outlier rows
        """
        if self.df is None:
            raise ValueError("No data loaded")

        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")

        col_data = self.df[column].dropna()

        if method == "iqr":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outlier_mask = (self.df[column] < lower_bound) | (
                self.df[column] > upper_bound
            )
            return self.df[outlier_mask]

        raise ValueError(f"Method {method} not implemented")


BOOKING_COLUMNS = {"num_passengers", "sales_channel", "trip_type", "booking_complete"}
PSEUDO_MISSING_VALUE = "(not set)"


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Detect whether a DataFrame contains booking or flight data.

    Returns:
        "booking" if booking-specific columns are present, "flight" otherwise
    """
    cols = set(df.columns)
    if BOOKING_COLUMNS.issubset(cols):
        return "booking"
    return "flight"


class FlightDataValidator:
    """Flight-specific data validation rules"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize FlightDataValidator

        Args:
            df: DataFrame containing flight data
        """
        self.df = df

    def validate_carrier_codes(self) -> pd.DataFrame:
        """
        Validate carrier codes (should be 2 characters)

        Returns:
            DataFrame of rows with invalid carrier codes
        """
        if "op_unique_carrier" not in self.df.columns:
            return pd.DataFrame()

        invalid_mask = self.df["op_unique_carrier"].str.len() != 2
        return self.df[invalid_mask]

    def validate_airport_codes(self) -> pd.DataFrame:
        """
        Validate airport codes (should be 3 characters)

        Returns:
            DataFrame of rows with invalid airport codes
        """
        invalid_rows = []

        for col in ["origin", "dest"]:
            if col in self.df.columns:
                invalid_mask = self.df[col].str.len() != 3
                invalid_rows.append(self.df[invalid_mask])

        if invalid_rows:
            return pd.concat(invalid_rows).drop_duplicates()

        return pd.DataFrame()

    def validate_dates(self) -> pd.DataFrame:
        """
        Validate date columns

        Returns:
            DataFrame of rows with invalid dates
        """
        if "fl_date" not in self.df.columns:
            return pd.DataFrame()

        # Check for dates outside reasonable range (2020-2030)
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2030-12-31")

        df_with_dates = self.df.copy()
        df_with_dates["fl_date"] = pd.to_datetime(
            df_with_dates["fl_date"], errors="coerce"
        )

        invalid_mask = (
            (df_with_dates["fl_date"] < min_date)
            | (df_with_dates["fl_date"] > max_date)
            | (df_with_dates["fl_date"].isna())
        )

        return self.df[invalid_mask]

    def validate_delay_logic(self) -> pd.DataFrame:
        """
        Validate delay logic - cancelled flights should have null delays

        Returns:
            DataFrame of rows with invalid delay logic
        """
        if "cancelled" not in self.df.columns:
            return pd.DataFrame()

        # Cancelled flights should have NaN delays
        invalid_mask = (self.df["cancelled"] == 1) & (
            self.df["dep_delay"].notna() | self.df["arr_delay"].notna()
        )

        return self.df[invalid_mask]

    def validate_origin_destination_different(self) -> pd.DataFrame:
        """
        Validate that origin and destination airports are different.
        Flags records where origin == dest (case-insensitive),
        including empty strings and NaN pairs.

        Returns:
            DataFrame of rows where origin and destination are the same
        """
        if "origin" not in self.df.columns or "dest" not in self.df.columns:
            return pd.DataFrame()

        origin = self.df["origin"].fillna("").str.upper()
        dest = self.df["dest"].fillna("").str.upper()

        invalid_mask = origin == dest
        return self.df[invalid_mask]

    def validate_distance_geodesic(self, threshold: float = 0.20) -> pd.DataFrame:
        """
        Validate that reported flight distances match expected distances
        calculated from airport coordinates using geodesic distance.

        Args:
            threshold: Maximum allowed relative difference (default 0.20 = 20%)

        Returns:
            DataFrame of flagged rows with expected_distance and actual_distance columns
        """
        required_cols = ["origin_lat", "origin_lon", "dest_lat", "dest_lon", "distance"]
        if not all(col in self.df.columns for col in required_cols):
            return pd.DataFrame()

        df = self.df.copy()

        # Skip rows with missing coordinate data
        coord_cols = ["origin_lat", "origin_lon", "dest_lat", "dest_lon"]
        has_coords = df[coord_cols].notna().all(axis=1)
        df_valid = df[has_coords].copy()

        if df_valid.empty:
            return pd.DataFrame()

        # Cache geodesic distances per unique route coordinate pair
        route_keys = df_valid[coord_cols].apply(tuple, axis=1)
        unique_routes = df_valid[coord_cols].drop_duplicates()
        distance_cache = {}
        for _, row in unique_routes.iterrows():
            key = (
                row["origin_lat"],
                row["origin_lon"],
                row["dest_lat"],
                row["dest_lon"],
            )
            distance_cache[key] = geodesic((key[0], key[1]), (key[2], key[3])).miles

        df_valid["expected_distance"] = route_keys.map(distance_cache)
        df_valid["actual_distance"] = df_valid["distance"]

        # Flag rows where difference exceeds threshold
        # Zero reported distance is always flagged when expected > 0
        # Note: when expected == 0 (same coordinates) and actual > 0, the row
        # is not flagged — this is intentional as co-located airports are rare
        # and warrant separate validation.
        expected = df_valid["expected_distance"]
        actual = df_valid["actual_distance"]
        flagged_mask = ((expected > 0) & (actual == 0)) | (
            (expected > 0) & ((actual - expected).abs() / expected > threshold)
        )

        return df_valid[flagged_mask]

    def validate_distance_positive(self) -> pd.DataFrame:
        """
        Validate that all distances are positive

        Returns:
            DataFrame of rows with non-positive distances
        """
        if "distance" not in self.df.columns:
            return pd.DataFrame()

        invalid_mask = self.df["distance"] <= 0
        return self.df[invalid_mask]


class CustomerBookingValidator:
    """Booking-specific data validation rules"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_booking_origin(self) -> pd.DataFrame:
        """Flag rows where booking_origin is '(not set)' (pseudo-missing data).

        Returns:
            DataFrame of rows with booking_origin == '(not set)'
        """
        if "booking_origin" not in self.df.columns:
            return pd.DataFrame()
        mask = self.df["booking_origin"].str.strip() == PSEUDO_MISSING_VALUE
        return self.df[mask]


class BookingCompletionAnalyzer:
    """Analyze booking completion rates across different dimensions"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overall_completion_rate(self) -> Dict[str, Any]:
        """Calculate overall booking completion rate."""
        total = len(self.df)
        if total == 0:
            return {"total_bookings": 0, "completed": 0, "completion_rate": 0.0}
        completed = int(self.df["booking_complete"].sum())
        rate = round((completed / total) * 100, 2)
        return {
            "total_bookings": total,
            "completed": completed,
            "completion_rate": rate,
        }

    def _group_completion(self, column: str) -> pd.DataFrame:
        """Helper to compute completion rate grouped by a column."""
        grouped = (
            self.df.groupby(column)
            .agg(
                total=("booking_complete", "count"),
                completed=("booking_complete", "sum"),
            )
            .reset_index()
        )
        grouped["completion_rate"] = (
            ((grouped["completed"] / grouped["total"]) * 100).fillna(0).round(2)
        )
        return grouped

    def completion_by_channel(self) -> pd.DataFrame:
        """Completion rate by sales channel."""
        return self._group_completion("sales_channel")

    def completion_by_trip_type(self) -> pd.DataFrame:
        """Completion rate by trip type."""
        return self._group_completion("trip_type")

    def completion_by_origin(self, top_n: int = 15) -> pd.DataFrame:
        """Completion rate by booking origin, sorted by volume, top N."""
        result = self._group_completion("booking_origin")
        result = result.sort_values("total", ascending=False).head(top_n)
        return result.reset_index(drop=True)

    def completion_by_extras(self) -> pd.DataFrame:
        """Completion rate by number of extras requested (0-3)."""
        extras_cols = [
            "wants_extra_baggage",
            "wants_preferred_seat",
            "wants_in_flight_meals",
        ]
        available = [c for c in extras_cols if c in self.df.columns]
        df = self.df.copy()
        df["num_extras"] = df[available].sum(axis=1).astype(int) if available else 0
        grouped = (
            df.groupby("num_extras")
            .agg(
                total=("booking_complete", "count"),
                completed=("booking_complete", "sum"),
            )
            .reset_index()
        )
        grouped["completion_rate"] = (
            ((grouped["completed"] / grouped["total"]) * 100).fillna(0).round(2)
        )
        return grouped

    def completion_by_flight_day(self) -> pd.DataFrame:
        """Completion rate by flight day."""
        return self._group_completion("flight_day")

    def volume_vs_completion(self) -> pd.DataFrame:
        """Volume percentage and completion rate by booking origin."""
        total_bookings = len(self.df)
        result = self._group_completion("booking_origin")
        result["volume_pct"] = round((result["total"] / total_bookings) * 100, 2)
        return result


class BookingOutlierDetector:
    """Detect outliers in booking numeric fields using thresholds and IQR"""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def flag_by_threshold(self, column: str, max_value: float) -> pd.DataFrame:
        """Flag rows where column value exceeds a business threshold.

        Args:
            column: Column name to check
            max_value: Maximum acceptable value

        Returns:
            DataFrame of rows exceeding the threshold
        """
        if column not in self.df.columns:
            return pd.DataFrame()
        mask = self.df[column] > max_value
        return self.df[mask]

    def iqr_outlier_summary(
        self, columns: List[str], threshold: float = 1.5
    ) -> pd.DataFrame:
        """Generate IQR-based outlier summary for given columns.

        Args:
            columns: List of numeric column names to analyze
            threshold: IQR multiplier (default 1.5)

        Returns:
            DataFrame with columns: column, outlier_count, outlier_pct,
            Q1, Q3, IQR, lower_bound, upper_bound
        """
        rows = []
        total = len(self.df)
        for col in columns:
            if col not in self.df.columns:
                continue
            col_data = self.df[col].dropna()
            if len(col_data) == 0:
                rows.append(
                    {
                        "column": col,
                        "outlier_count": 0,
                        "outlier_pct": 0.0,
                        "Q1": 0.0,
                        "Q3": 0.0,
                        "IQR": 0.0,
                        "lower_bound": 0.0,
                        "upper_bound": 0.0,
                    }
                )
                continue
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_count = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
            outlier_pct = round((outlier_count / total) * 100, 2) if total > 0 else 0.0
            rows.append(
                {
                    "column": col,
                    "outlier_count": outlier_count,
                    "outlier_pct": outlier_pct,
                    "Q1": round(q1, 2),
                    "Q3": round(q3, 2),
                    "IQR": round(iqr, 2),
                    "lower_bound": round(lower, 2),
                    "upper_bound": round(upper, 2),
                }
            )
        return pd.DataFrame(rows)


class BaseDataQualityReport:
    """Base class with shared report generation logic"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.checker = DataQualityChecker()
        self.checker.df = df

    def _generate_overview(self) -> Dict[str, Any]:
        """Generate overview statistics"""
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "memory_usage": int(self.df.memory_usage(deep=True).sum()),
        }

    def _generate_null_analysis(self) -> Dict[str, float]:
        return self.checker.calculate_null_percentage()

    def _generate_duplicates(self) -> Dict[str, Any]:
        total_rows = len(self.df)
        duplicate_count = int(self.checker.check_duplicates())
        return {
            "total_duplicates": duplicate_count,
            "duplicate_percentage": round((duplicate_count / total_rows) * 100, 2)
            if total_rows > 0
            else 0.0,
        }

    def _generate_statistics(self) -> Dict[str, Any]:
        statistics: Dict[str, Any] = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                statistics[col] = self.checker.calculate_statistics(col)
            except Exception as e:
                statistics[col] = {"error": str(e)}
        return statistics


class BookingDataQualityReport(BaseDataQualityReport):
    """Generate data quality reports for customer booking data"""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.validator = CustomerBookingValidator(df)

    def generate(self) -> Dict[str, Any]:
        """Generate complete booking data quality report."""
        total_rows = len(self.df)
        duplicates = self._generate_duplicates()

        completion_rate = 0.0
        if "booking_complete" in self.df.columns and total_rows > 0:
            completion_rate = round(
                (self.df["booking_complete"].sum() / total_rows) * 100, 2
            )

        not_set_count = len(self.validator.validate_booking_origin())

        overview = self._generate_overview()
        overview["completion_rate"] = completion_rate
        overview["duplicate_count"] = duplicates["total_duplicates"]

        return {
            "generated_at": datetime.now().isoformat(),
            "dataset_type": "booking",
            "overview": overview,
            "null_analysis": self._generate_null_analysis(),
            "duplicates": duplicates,
            "statistics": self._generate_statistics(),
            "not_set_booking_origin": not_set_count,
        }


class DataQualityReport(BaseDataQualityReport):
    """Generate comprehensive data quality reports"""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.validator = FlightDataValidator(df)

    def _generate_overview(self) -> Dict[str, Any]:
        """Generate overview statistics with column list"""
        overview = super()._generate_overview()
        overview["column_list"] = list(self.df.columns)
        return overview

    def generate_route_validation_details(
        self,
        same_airport: Optional[pd.DataFrame] = None,
        distance_mismatches: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate detailed route validation results with issue type and severity.

        Args:
            same_airport: Pre-computed same-airport violations. If None, runs validation.
            distance_mismatches: Pre-computed distance mismatches. If None, runs validation.

        Returns:
            DataFrame with columns: flight_date, carrier, origin, dest,
            issue_type, severity, details
        """
        rows = []

        if same_airport is None:
            same_airport = self.validator.validate_origin_destination_different()
        if distance_mismatches is None:
            distance_mismatches = self.validator.validate_distance_geodesic()

        # Same airport violations (High severity)
        for _, row in same_airport.iterrows():
            rows.append(
                {
                    "flight_date": row.get("fl_date", ""),
                    "carrier": row.get("op_unique_carrier", ""),
                    "origin": row.get("origin", ""),
                    "dest": row.get("dest", ""),
                    "issue_type": "Same Airport",
                    "severity": "High",
                    "details": f"Origin and destination are the same: {row.get('origin', '')}",
                }
            )

        # Distance mismatches (Medium severity)
        for _, row in distance_mismatches.iterrows():
            expected = row.get("expected_distance", 0)
            actual = row.get("actual_distance", 0)
            rows.append(
                {
                    "flight_date": row.get("fl_date", ""),
                    "carrier": row.get("op_unique_carrier", ""),
                    "origin": row.get("origin", ""),
                    "dest": row.get("dest", ""),
                    "issue_type": "Distance Mismatch",
                    "severity": "Medium",
                    "details": f"Expected: {expected:.0f} mi, Actual: {actual:.0f} mi",
                }
            )

        columns = [
            "flight_date",
            "carrier",
            "origin",
            "dest",
            "issue_type",
            "severity",
            "details",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)

    def generate(self) -> Dict[str, Any]:
        """
        Generate complete data quality report

        Returns:
            Dictionary containing all report sections
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "overview": self._generate_overview(),
            "null_analysis": self._generate_null_analysis(),
            "duplicates": self._generate_duplicates(),
            "statistics": self._generate_statistics(),
            "validations": {},
        }

        # Run flight-specific validations (compute once, reuse for details)
        same_origin_dest_df = self.validator.validate_origin_destination_different()
        distance_mismatches_df = self.validator.validate_distance_geodesic()

        report["validations"] = {
            "invalid_carrier_codes": len(self.validator.validate_carrier_codes()),
            "invalid_airport_codes": len(self.validator.validate_airport_codes()),
            "invalid_dates": len(self.validator.validate_dates()),
            "invalid_delay_logic": len(self.validator.validate_delay_logic()),
            "invalid_distances": len(self.validator.validate_distance_positive()),
            "same_origin_destination": len(same_origin_dest_df),
            "distance_mismatches": len(distance_mismatches_df),
        }

        # Add detailed route validation results (reuse pre-computed DataFrames)
        route_details = self.generate_route_validation_details(
            same_airport=same_origin_dest_df,
            distance_mismatches=distance_mismatches_df,
        )
        report["route_validation_details"] = route_details.to_dict(orient="records")

        return report

    def export_json(self, report: Dict[str, Any], filepath: str):
        """Export report to JSON file"""
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    def export_html(self, report: Dict[str, Any], filepath: str):
        """Export report to HTML file"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd;
                     padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
                .metric-label {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <p><strong>Generated:</strong> {generated_at}</p>
            
            <h2>Overview</h2>
            <div class="metric">
                <div class="metric-value">{total_rows:,}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_columns}</div>
                <div class="metric-label">Total Columns</div>
            </div>
            <div class="metric">
                <div class="metric-value">{memory_mb:.2f} MB</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            
            <h2>Null Analysis</h2>
            <table>
                <tr><th>Column</th><th>Null %</th></tr>
                {null_rows}
            </table>
            
            <h2>Duplicates</h2>
            <p><strong>Total Duplicates:</strong>
            {total_duplicates:,} ({duplicate_percentage}%)</p>
            
            <h2>Validation Results</h2>
            <table>
                <tr><th>Validation</th><th>Issues Found</th></tr>
                {validation_rows}
            </table>
        </body>
        </html>
        """

        # Generate table rows for null analysis
        null_rows = "\n".join(
            [
                f"<tr><td>{col}</td><td>{pct}%</td></tr>"
                for col, pct in report["null_analysis"].items()
            ]
        )

        # Generate validation rows
        validation_rows = "\n".join(
            [
                f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
                for key, value in report["validations"].items()
            ]
        )

        html_content = html_template.format(
            generated_at=report["generated_at"],
            total_rows=report["overview"]["total_rows"],
            total_columns=report["overview"]["total_columns"],
            memory_mb=report["overview"]["memory_usage"] / (1024 * 1024),
            null_rows=null_rows,
            total_duplicates=report["duplicates"]["total_duplicates"],
            duplicate_percentage=report["duplicates"]["duplicate_percentage"],
            validation_rows=validation_rows,
        )

        with open(filepath, "w") as f:
            f.write(html_content)


def run_quality_check(filepath: str, output_dir: str = "./reports") -> Dict[str, Any]:
    """
    Run complete data quality check pipeline

    Args:
        filepath: Path to CSV file to analyze
        output_dir: Directory to save reports

    Returns:
        Data quality report dictionary
    """
    # Load data
    checker = DataQualityChecker(filepath)

    # Generate report
    report_gen = DataQualityReport(checker.df)
    report = report_gen.generate()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export reports
    report_gen.export_json(report, str(output_path / "data_quality_report.json"))
    report_gen.export_html(report, str(output_path / "data_quality_report.html"))

    return report
