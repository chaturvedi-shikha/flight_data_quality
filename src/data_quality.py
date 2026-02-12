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
            'count': int(len(col_data)),
            'mean': float(col_data.mean()) if len(col_data) > 0 else None,
            'median': float(col_data.median()) if len(col_data) > 0 else None,
            'std': float(col_data.std()) if len(col_data) > 0 else None,
            'min': float(col_data.min()) if len(col_data) > 0 else None,
            'max': float(col_data.max()) if len(col_data) > 0 else None
        }
    
    def detect_outliers(self, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
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
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
            return self.df[outlier_mask]
        
        raise ValueError(f"Method {method} not implemented")


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
        if 'op_unique_carrier' not in self.df.columns:
            return pd.DataFrame()
        
        invalid_mask = self.df['op_unique_carrier'].str.len() != 2
        return self.df[invalid_mask]
    
    def validate_airport_codes(self) -> pd.DataFrame:
        """
        Validate airport codes (should be 3 characters)
        
        Returns:
            DataFrame of rows with invalid airport codes
        """
        invalid_rows = []
        
        for col in ['origin', 'dest']:
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
        if 'fl_date' not in self.df.columns:
            return pd.DataFrame()
        
        # Check for dates outside reasonable range (2020-2030)
        min_date = pd.Timestamp('2020-01-01')
        max_date = pd.Timestamp('2030-12-31')
        
        df_with_dates = self.df.copy()
        df_with_dates['fl_date'] = pd.to_datetime(df_with_dates['fl_date'], errors='coerce')
        
        invalid_mask = (
            (df_with_dates['fl_date'] < min_date) |
            (df_with_dates['fl_date'] > max_date) |
            (df_with_dates['fl_date'].isna())
        )
        
        return self.df[invalid_mask]
    
    def validate_delay_logic(self) -> pd.DataFrame:
        """
        Validate delay logic - cancelled flights should have null delays
        
        Returns:
            DataFrame of rows with invalid delay logic
        """
        if 'cancelled' not in self.df.columns:
            return pd.DataFrame()
        
        # Cancelled flights should have NaN delays
        invalid_mask = (
            (self.df['cancelled'] == 1) &
            (self.df['dep_delay'].notna() | self.df['arr_delay'].notna())
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
        if 'origin' not in self.df.columns or 'dest' not in self.df.columns:
            return pd.DataFrame()

        origin = self.df['origin'].fillna('').str.upper()
        dest = self.df['dest'].fillna('').str.upper()

        invalid_mask = origin == dest
        return self.df[invalid_mask]

    def validate_distance_haversine(self, threshold: float = 0.20) -> pd.DataFrame:
        """
        Validate that reported flight distances match expected distances
        calculated from airport coordinates using geodesic distance.

        Args:
            threshold: Maximum allowed relative difference (default 0.20 = 20%)

        Returns:
            DataFrame of flagged rows with expected_distance and actual_distance columns
        """
        required_cols = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon', 'distance']
        if not all(col in self.df.columns for col in required_cols):
            return pd.DataFrame()

        df = self.df.copy()

        # Skip rows with missing coordinate data
        coord_cols = ['origin_lat', 'origin_lon', 'dest_lat', 'dest_lon']
        has_coords = df[coord_cols].notna().all(axis=1)
        df_valid = df[has_coords].copy()

        if df_valid.empty:
            return pd.DataFrame()

        # Calculate expected distance using geodesic
        df_valid['expected_distance'] = df_valid.apply(
            lambda row: geodesic(
                (row['origin_lat'], row['origin_lon']),
                (row['dest_lat'], row['dest_lon'])
            ).miles,
            axis=1
        )
        df_valid['actual_distance'] = df_valid['distance']

        # Flag rows where difference exceeds threshold
        # Zero reported distance is always flagged when expected > 0
        expected = df_valid['expected_distance']
        actual = df_valid['actual_distance']
        flagged_mask = (
            ((expected > 0) & (actual == 0)) |
            ((expected > 0) & ((actual - expected).abs() / expected > threshold))
        )

        return df_valid[flagged_mask]

    def validate_distance_positive(self) -> pd.DataFrame:
        """
        Validate that all distances are positive
        
        Returns:
            DataFrame of rows with non-positive distances
        """
        if 'distance' not in self.df.columns:
            return pd.DataFrame()
        
        invalid_mask = self.df['distance'] <= 0
        return self.df[invalid_mask]


class DataQualityReport:
    """Generate comprehensive data quality reports"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataQualityReport
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.checker = DataQualityChecker()
        self.checker.df = df
        self.validator = FlightDataValidator(df)
    
    def _generate_overview(self) -> Dict[str, Any]:
        """Generate overview statistics"""
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage': int(self.df.memory_usage(deep=True).sum()),
            'column_list': list(self.df.columns)
        }
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate complete data quality report
        
        Returns:
            Dictionary containing all report sections
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'overview': self._generate_overview(),
            'null_analysis': self.checker.calculate_null_percentage(),
            'duplicates': {
                'total_duplicates': int(self.checker.check_duplicates()),
                'duplicate_percentage': round(
                    (self.checker.check_duplicates() / len(self.df)) * 100, 2
                )
            },
            'statistics': {},
            'validations': {}
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                report['statistics'][col] = self.checker.calculate_statistics(col)
            except Exception as e:
                report['statistics'][col] = {'error': str(e)}
        
        # Run flight-specific validations
        report['validations'] = {
            'invalid_carrier_codes': len(self.validator.validate_carrier_codes()),
            'invalid_airport_codes': len(self.validator.validate_airport_codes()),
            'invalid_dates': len(self.validator.validate_dates()),
            'invalid_delay_logic': len(self.validator.validate_delay_logic()),
            'invalid_distances': len(self.validator.validate_distance_positive()),
            'same_origin_destination': len(self.validator.validate_origin_destination_different()),
            'distance_mismatches': len(self.validator.validate_distance_haversine())
        }
        
        return report
    
    def export_json(self, report: Dict[str, Any], filepath: str):
        """Export report to JSON file"""
        with open(filepath, 'w') as f:
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
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
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
            <p><strong>Total Duplicates:</strong> {total_duplicates:,} ({duplicate_percentage}%)</p>
            
            <h2>Validation Results</h2>
            <table>
                <tr><th>Validation</th><th>Issues Found</th></tr>
                {validation_rows}
            </table>
        </body>
        </html>
        """
        
        # Generate table rows for null analysis
        null_rows = '\n'.join([
            f"<tr><td>{col}</td><td>{pct}%</td></tr>"
            for col, pct in report['null_analysis'].items()
        ])
        
        # Generate validation rows
        validation_rows = '\n'.join([
            f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
            for key, value in report['validations'].items()
        ])
        
        html_content = html_template.format(
            generated_at=report['generated_at'],
            total_rows=report['overview']['total_rows'],
            total_columns=report['overview']['total_columns'],
            memory_mb=report['overview']['memory_usage'] / (1024 * 1024),
            null_rows=null_rows,
            total_duplicates=report['duplicates']['total_duplicates'],
            duplicate_percentage=report['duplicates']['duplicate_percentage'],
            validation_rows=validation_rows
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)


def run_quality_check(filepath: str, output_dir: str = './reports') -> Dict[str, Any]:
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
    report_gen.export_json(report, str(output_path / 'data_quality_report.json'))
    report_gen.export_html(report, str(output_path / 'data_quality_report.html'))
    
    return report
