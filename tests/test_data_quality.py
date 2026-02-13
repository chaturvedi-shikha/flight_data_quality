"""
Test suite for Flight Data Quality Checks
Following TDD approach - tests written before implementation
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_flight_data():
    """Create sample flight data for testing"""
    return pd.DataFrame(
        {
            "year": [2024, 2024, 2024],
            "month": [1, 1, 1],
            "fl_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "op_unique_carrier": ["AA", "DL", "UA"],
            "origin": ["JFK", "LAX", "ORD"],
            "dest": ["LAX", "JFK", "SFO"],
            "dep_delay": [10.0, -5.0, np.nan],
            "arr_delay": [15.0, -3.0, np.nan],
            "cancelled": [0, 0, 1],
            "distance": [2475.0, 2475.0, 1846.0],
        }
    )


@pytest.fixture
def sample_data_with_nulls():
    """Create data with various null patterns"""
    return pd.DataFrame(
        {
            "col_no_nulls": [1, 2, 3, 4, 5],
            "col_some_nulls": [1, np.nan, 3, np.nan, 5],
            "col_all_nulls": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "col_string_nulls": ["A", None, "C", "", "E"],
        }
    )


class TestDataQualityChecker:
    """Test suite for DataQualityChecker class"""

    def test_load_data(self, tmp_path):
        """Test data loading from CSV"""
        from src.data_quality import DataQualityChecker

        # Create temp CSV
        test_data = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        test_file = tmp_path / "test.csv"
        test_data.to_csv(test_file, index=False)

        checker = DataQualityChecker(str(test_file))
        assert checker.df is not None
        assert len(checker.df) == 2
        assert list(checker.df.columns) == ["col1", "col2"]

    def test_calculate_null_percentage(self, sample_data_with_nulls):
        """Test null percentage calculation"""
        from src.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        checker.df = sample_data_with_nulls

        null_stats = checker.calculate_null_percentage()

        assert null_stats["col_no_nulls"] == 0.0
        assert null_stats["col_some_nulls"] == 40.0
        assert null_stats["col_all_nulls"] == 100.0

    def test_check_duplicates(self):
        """Test duplicate detection"""
        from src.data_quality import DataQualityChecker

        data = pd.DataFrame({"id": [1, 2, 2, 3], "value": ["A", "B", "B", "C"]})

        checker = DataQualityChecker()
        checker.df = data

        duplicate_count = checker.check_duplicates()
        assert duplicate_count == 1  # One duplicate row

    def test_check_duplicates_by_column(self):
        """Test duplicate detection by specific column"""
        from src.data_quality import DataQualityChecker

        data = pd.DataFrame({"id": [1, 2, 2, 3], "value": ["A", "B", "C", "D"]})

        checker = DataQualityChecker()
        checker.df = data

        duplicate_count = checker.check_duplicates(subset=["id"])
        assert duplicate_count == 1  # One duplicate ID

    def test_calculate_statistics(self, sample_flight_data):
        """Test statistical summary calculation"""
        from src.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        checker.df = sample_flight_data

        stats = checker.calculate_statistics("dep_delay")

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == 2  # Only 2 non-null values

    def test_detect_outliers(self):
        """Test outlier detection using IQR method"""
        from src.data_quality import DataQualityChecker

        data = pd.DataFrame({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200]})

        checker = DataQualityChecker()
        checker.df = data

        outliers = checker.detect_outliers("values", method="iqr")
        assert len(outliers) == 2
        assert 100 in outliers["values"].values
        assert 200 in outliers["values"].values


class TestFlightDataValidator:
    """Test suite for flight-specific validations"""

    def test_validate_carrier_codes(self, sample_flight_data):
        """Test carrier code validation (should be 2 characters)"""
        from src.data_quality import FlightDataValidator

        validator = FlightDataValidator(sample_flight_data)
        invalid_carriers = validator.validate_carrier_codes()

        assert len(invalid_carriers) == 0  # All carriers are valid

    def test_validate_airport_codes(self, sample_flight_data):
        """Test airport code validation (should be 3 characters)"""
        from src.data_quality import FlightDataValidator

        validator = FlightDataValidator(sample_flight_data)
        invalid_airports = validator.validate_airport_codes()

        assert len(invalid_airports) == 0  # All airport codes are valid

    def test_validate_dates(self, sample_flight_data):
        """Test date validation"""
        from src.data_quality import FlightDataValidator

        validator = FlightDataValidator(sample_flight_data)
        invalid_dates = validator.validate_dates()

        assert len(invalid_dates) == 0  # All dates are valid

    def test_validate_delay_logic(self):
        """Test delay validation - cancelled flights should have null delays"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {
                "cancelled": [0, 1],
                "dep_delay": [10.0, 5.0],  # Cancelled flight shouldn't have delay
                "arr_delay": [15.0, 8.0],
            }
        )

        validator = FlightDataValidator(data)
        invalid_delays = validator.validate_delay_logic()

        assert len(invalid_delays) == 1  # One cancelled flight with delay

    def test_validate_distance_positive(self, sample_flight_data):
        """Test that all distances are positive"""
        from src.data_quality import FlightDataValidator

        validator = FlightDataValidator(sample_flight_data)
        invalid_distances = validator.validate_distance_positive()

        assert len(invalid_distances) == 0  # All distances are positive

    def test_validate_origin_dest_different_valid(self, sample_flight_data):
        """Test valid routes where origin != destination"""
        from src.data_quality import FlightDataValidator

        validator = FlightDataValidator(sample_flight_data)
        invalid_routes = validator.validate_origin_destination_different()

        assert len(invalid_routes) == 0  # All routes are valid

    def test_validate_origin_dest_different_same_airport(self):
        """Test detection of same origin-destination routes"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {"origin": ["JFK", "LAX", "ORD"], "dest": ["JFK", "SFO", "ORD"]}
        )

        validator = FlightDataValidator(data)
        invalid_routes = validator.validate_origin_destination_different()

        assert len(invalid_routes) == 2  # JFK→JFK and ORD→ORD

    def test_validate_origin_dest_different_null_values(self):
        """Test handling of null values in origin/dest"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {"origin": [None, "LAX", np.nan], "dest": ["JFK", "SFO", np.nan]}
        )

        validator = FlightDataValidator(data)
        invalid_routes = validator.validate_origin_destination_different()

        assert len(invalid_routes) == 1  # NaN→NaN should be flagged

    def test_validate_origin_dest_different_empty_strings(self):
        """Test handling of empty strings in origin/dest"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame({"origin": ["", "LAX", "JFK"], "dest": ["", "SFO", "LAX"]})

        validator = FlightDataValidator(data)
        invalid_routes = validator.validate_origin_destination_different()

        assert len(invalid_routes) == 1  # ""→"" should be flagged

    def test_validate_origin_dest_different_case_sensitivity(self):
        """Test case-insensitive comparison"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame({"origin": ["JFK", "lax"], "dest": ["jfk", "SFO"]})

        validator = FlightDataValidator(data)
        invalid_routes = validator.validate_origin_destination_different()

        assert len(invalid_routes) == 1  # JFK→jfk should be flagged

    def test_validate_distance_accurate(self):
        """Test accurate distance: JFK-LAX ~2,475 mi matches coords"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {
                "origin": ["JFK"],
                "dest": ["LAX"],
                "origin_lat": [40.6413],
                "origin_lon": [-73.7781],
                "dest_lat": [33.9425],
                "dest_lon": [-118.4081],
                "distance": [2475.0],
            }
        )

        validator = FlightDataValidator(data)
        flagged = validator.validate_distance_geodesic()

        assert len(flagged) == 0  # Accurate distance should not be flagged

    def test_validate_distance_minor_variance(self):
        """Test minor variance: distance off by 10% (should pass with 20% threshold)"""
        from src.data_quality import FlightDataValidator

        # JFK→LAX expected ~2,475 mi, report 2,722 (10% over)
        data = pd.DataFrame(
            {
                "origin": ["JFK"],
                "dest": ["LAX"],
                "origin_lat": [40.6413],
                "origin_lon": [-73.7781],
                "dest_lat": [33.9425],
                "dest_lon": [-118.4081],
                "distance": [2722.0],
            }
        )

        validator = FlightDataValidator(data)
        flagged = validator.validate_distance_geodesic()

        assert len(flagged) == 0  # 10% variance within 20% threshold

    def test_validate_distance_major_variance(self):
        """Test major variance: distance off by 50% (should flag)"""
        from src.data_quality import FlightDataValidator

        # JFK→LAX expected ~2,475 mi, report 3,712 (50% over)
        data = pd.DataFrame(
            {
                "origin": ["JFK"],
                "dest": ["LAX"],
                "origin_lat": [40.6413],
                "origin_lon": [-73.7781],
                "dest_lat": [33.9425],
                "dest_lon": [-118.4081],
                "distance": [3712.0],
            }
        )

        validator = FlightDataValidator(data)
        flagged = validator.validate_distance_geodesic()

        assert len(flagged) == 1  # 50% variance exceeds 20% threshold
        assert "expected_distance" in flagged.columns

    def test_validate_distance_missing_coordinates(self):
        """Test missing coordinates: lat/lon is null (should skip validation)"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {
                "origin": ["JFK", "LAX"],
                "dest": ["LAX", "SFO"],
                "origin_lat": [40.6413, np.nan],
                "origin_lon": [-73.7781, np.nan],
                "dest_lat": [33.9425, np.nan],
                "dest_lon": [-118.4081, np.nan],
                "distance": [2475.0, 500.0],
            }
        )

        validator = FlightDataValidator(data)
        flagged = validator.validate_distance_geodesic()

        assert len(flagged) == 0  # Missing coords skipped, valid row passes

    def test_validate_distance_same_coordinates(self):
        """Test edge case: expected dist 0, actual > 0 — not flagged"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {
                "origin": ["AAA"],
                "dest": ["BBB"],
                "origin_lat": [40.6413],
                "origin_lon": [-73.7781],
                "dest_lat": [40.6413],
                "dest_lon": [-73.7781],
                "distance": [100.0],
            }
        )

        validator = FlightDataValidator(data)
        flagged = validator.validate_distance_geodesic()

        # Intentionally not flagged; co-located airports are separate
        assert len(flagged) == 0

    def test_validate_distance_zero(self):
        """Test zero distance: reported distance = 0 (should flag)"""
        from src.data_quality import FlightDataValidator

        data = pd.DataFrame(
            {
                "origin": ["JFK"],
                "dest": ["LAX"],
                "origin_lat": [40.6413],
                "origin_lon": [-73.7781],
                "dest_lat": [33.9425],
                "dest_lon": [-118.4081],
                "distance": [0.0],
            }
        )

        validator = FlightDataValidator(data)
        flagged = validator.validate_distance_geodesic()

        assert len(flagged) == 1  # Zero distance should be flagged


class TestDataQualityReport:
    """Test suite for data quality reporting"""

    def test_generate_report_structure(self, sample_flight_data):
        """Test that report generates all required sections"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(sample_flight_data)
        report_dict = report.generate()

        assert "overview" in report_dict
        assert "null_analysis" in report_dict
        assert "duplicates" in report_dict
        assert "statistics" in report_dict
        assert "validations" in report_dict

    def test_report_overview_metrics(self, sample_flight_data):
        """Test overview section contains key metrics"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(sample_flight_data)
        overview = report._generate_overview()

        assert "total_rows" in overview
        assert "total_columns" in overview
        assert "memory_usage" in overview
        assert overview["total_rows"] == 3
        assert overview["total_columns"] == 10

    def test_export_to_json(self, sample_flight_data, tmp_path):
        """Test exporting report to JSON"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(sample_flight_data)
        report_dict = report.generate()

        output_file = tmp_path / "report.json"
        report.export_json(report_dict, str(output_file))

        assert output_file.exists()

    def test_export_to_html(self, sample_flight_data, tmp_path):
        """Test exporting report to HTML"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(sample_flight_data)
        report_dict = report.generate()

        output_file = tmp_path / "report.html"
        report.export_html(report_dict, str(output_file))

        assert output_file.exists()

        # Check HTML contains key elements
        html_content = output_file.read_text()
        assert "<html>" in html_content
        assert "Data Quality Report" in html_content


class TestRouteValidationDetails:
    """Test suite for route validation detail generation"""

    @pytest.fixture
    def data_with_same_airport(self):
        """Provides a DataFrame with a same-airport issue."""
        return pd.DataFrame(
            {
                "fl_date": ["2024-01-01", "2024-01-02"],
                "op_unique_carrier": ["AA", "DL"],
                "origin": ["JFK", "LAX"],
                "dest": ["JFK", "SFO"],
                "dep_delay": [10.0, -5.0],
                "arr_delay": [15.0, -3.0],
                "cancelled": [0, 0],
                "distance": [2475.0, 350.0],
            }
        )

    @pytest.fixture
    def valid_flight_data(self):
        """Provides a DataFrame with no route validation issues."""
        return pd.DataFrame(
            {
                "fl_date": ["2024-01-01"],
                "op_unique_carrier": ["AA"],
                "origin": ["JFK"],
                "dest": ["LAX"],
                "dep_delay": [10.0],
                "arr_delay": [15.0],
                "cancelled": [0],
                "distance": [2475.0],
            }
        )

    def test_generate_route_validation_details_structure(self, data_with_same_airport):
        """Test that route validation details returns expected structure"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(data_with_same_airport)
        details = report.generate_route_validation_details()

        assert isinstance(details, pd.DataFrame)
        expected_cols = {
            "flight_date",
            "carrier",
            "origin",
            "dest",
            "issue_type",
            "severity",
            "details",
        }
        assert expected_cols.issubset(set(details.columns))

    def test_same_airport_flagged_as_high_severity(self, data_with_same_airport):
        """Test that same-airport violations are flagged with high severity"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(data_with_same_airport)
        details = report.generate_route_validation_details()

        same_airport = details[details["issue_type"] == "Same Airport"]
        assert len(same_airport) == 1
        assert same_airport.iloc[0]["severity"] == "High"
        assert same_airport.iloc[0]["origin"] == "JFK"

    def test_distance_mismatch_flagged_as_medium_severity(self):
        """Test that distance mismatches are flagged with medium severity"""
        from src.data_quality import DataQualityReport

        data = pd.DataFrame(
            {
                "fl_date": ["2024-01-01"],
                "op_unique_carrier": ["AA"],
                "origin": ["JFK"],
                "dest": ["LAX"],
                "origin_lat": [40.6413],
                "origin_lon": [-73.7781],
                "dest_lat": [33.9425],
                "dest_lon": [-118.4081],
                "dep_delay": [10.0],
                "arr_delay": [15.0],
                "cancelled": [0],
                "distance": [5000.0],  # Way off from ~2475
            }
        )

        report = DataQualityReport(data)
        details = report.generate_route_validation_details()

        distance_issues = details[details["issue_type"] == "Distance Mismatch"]
        assert len(distance_issues) == 1
        assert distance_issues.iloc[0]["severity"] == "Medium"

    def test_no_issues_returns_empty_dataframe(self, valid_flight_data):
        """Test that valid data returns empty DataFrame"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(valid_flight_data)
        details = report.generate_route_validation_details()

        assert len(details) == 0

    def test_report_includes_route_validation_details(self, valid_flight_data):
        """Test that generate() report includes route_validation_details key"""
        from src.data_quality import DataQualityReport

        report = DataQualityReport(valid_flight_data)
        report_dict = report.generate()

        assert "route_validation_details" in report_dict


class TestIntegration:
    """Integration tests for complete workflow"""

    def test_end_to_end_quality_check(self, tmp_path):
        """Test complete data quality pipeline"""
        from src.data_quality import run_quality_check

        # Create test data
        test_data = pd.DataFrame(
            {
                "year": [2024] * 5,
                "op_unique_carrier": ["AA", "DL", "UA", "AA", "DL"],
                "fl_date": ["2024-01-01"] * 5,
                "origin": ["JFK", "LAX", "ORD", "JFK", "LAX"],
                "dest": ["LAX", "JFK", "SFO", "LAX", "JFK"],
                "dep_delay": [10.0, -5.0, 0.0, np.nan, 15.0],
                "arr_delay": [15.0, -3.0, 5.0, np.nan, 20.0],
                "distance": [2475, 2475, 1846, 2475, 2475],
                "cancelled": [0, 0, 0, 1, 0],
            }
        )

        test_file = tmp_path / "test_flights.csv"
        test_data.to_csv(test_file, index=False)

        # Run quality check
        report = run_quality_check(str(test_file), output_dir=str(tmp_path))

        assert report is not None
        assert (tmp_path / "data_quality_report.json").exists()
