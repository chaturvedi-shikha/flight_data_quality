# Flight Data Quality Analysis

A comprehensive data quality analysis tool for flight data with TDD (Test-Driven Development) approach and interactive UI dashboard.

## Features

- **Comprehensive Data Quality Checks**
  - Missing data analysis
  - Duplicate detection
  - Statistical summaries
  - Outlier detection
  - Flight-specific validations (carrier codes, airport codes, dates, delays)

- **Test-Driven Development**
  - Full pytest test suite
  - 15+ test cases covering all functionality
  - Fixtures for repeatable testing
  - Integration tests for end-to-end workflows

- **Interactive Dashboard**
  - Streamlit-based web UI
  - Real-time data quality metrics
  - Interactive visualizations with Plotly
  - Download reports in JSON/CSV/HTML formats

- **Automated Reporting**
  - JSON exports for API integration
  - HTML reports for stakeholders
  - Customizable output formats

## Project Structure

```
flight-data-quality/
├── data/                           # Data files
│   ├── flight_data_2024_sample.csv
│   └── flight_data_2024_data_dictionary.csv
├── src/                            # Source code
│   └── data_quality.py            # Core analysis module
├── tests/                          # Test suite
│   └── test_data_quality.py       # TDD tests
├── ui/                            # UI dashboard
│   └── dashboard.py               # Streamlit app
├── reports/                        # Generated reports
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/chaturvedi-shika/flight-data-quality.git
cd flight-data-quality
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running Tests (TDD Approach)

```bash
# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=src --cov-report=term-missing

# Run specific test class
pytest -v tests/test_data_quality.py::TestDataQualityChecker
```

### Command Line Analysis

```python
from src.data_quality import run_quality_check

# Analyze a dataset
report = run_quality_check(
    filepath='data/flight_data_2024_sample.csv',
    output_dir='reports'
)

# Reports will be generated in reports/ directory:
# - data_quality_report.json
# - data_quality_report.html
```

### Interactive Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run ui/dashboard.py
```

Then open your browser to `http://localhost:8501`

**Dashboard Features:**
- Upload custom CSV files or use sample data
- View real-time data quality metrics
- Explore interactive visualizations
- Download reports in multiple formats

## Data Quality Checks

### 1. Missing Data Analysis
- Calculates null percentage for each column
- Visualizes missing data patterns
- Identifies columns with high missing rates

### 2. Duplicate Detection
- Finds exact duplicate rows
- Checks duplicates by specific columns
- Calculates duplicate percentage

### 3. Statistical Analysis
- Mean, median, standard deviation
- Min/max values
- Distribution analysis

### 4. Outlier Detection
- IQR (Interquartile Range) method
- Configurable threshold
- Identifies anomalous values

### 5. Flight-Specific Validations
- Carrier code format (2 characters)
- Airport code format (3 characters)
- Date range validation (2020-2030)
- Delay logic (cancelled flights shouldn't have delays)
- Distance validation (must be positive)

## Example Output

### Console Output
```
Data Quality Report Generated
============================
Total Rows: 10,000
Total Columns: 36
Duplicate Records: 12 (0.12%)

Null Analysis:
- dep_delay: 1.31%
- arr_delay: 1.61%
- cancellation_code: 98.64%

Validation Issues:
- Invalid carrier codes: 0
- Invalid airport codes: 5
- Invalid dates: 0
```

### HTML Report
Generated HTML includes:
- Overview metrics with visual cards
- Null analysis table with color coding
- Validation results summary
- Statistical summaries

## Dataset

**Flight Data 2024**
- Source: US Department of Transportation
- Records: ~10,000 (sample), 1M+ (full dataset)
- Columns: 36 (includes delays, cancellations, carriers, routes)

**Key Columns:**
- `fl_date`: Flight date
- `op_unique_carrier`: Operating carrier code
- `origin`, `dest`: Airport codes
- `dep_delay`, `arr_delay`: Departure/arrival delays
- `cancelled`: Cancellation flag
- `distance`: Flight distance in miles

## Development

### TDD Workflow

1. **Write tests first** (tests/test_data_quality.py)
2. **Run tests** (they should fail initially)
3. **Implement functionality** (src/data_quality.py)
4. **Run tests again** (they should pass)
5. **Refactor** and repeat

### Adding New Checks

1. Add test in `tests/test_data_quality.py`:
```python
def test_new_validation(self, sample_data):
    from src.data_quality import FlightDataValidator
    
    validator = FlightDataValidator(sample_data)
    result = validator.validate_new_check()
    
    assert len(result) == expected_value
```

2. Implement in `src/data_quality.py`:
```python
def validate_new_check(self) -> pd.DataFrame:
    """Your validation logic"""
    # Implementation
    return invalid_rows
```

3. Run tests:
```bash
pytest -v tests/test_data_quality.py::TestFlightDataValidator::test_new_validation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-check`)
3. Write tests for your feature (TDD)
4. Implement the feature
5. Ensure all tests pass (`pytest -v`)
6. Commit your changes (`git commit -am 'Add new validation'`)
7. Push to the branch (`git push origin feature/new-check`)
8. Create a Pull Request

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **streamlit**: Dashboard UI
- **plotly**: Interactive visualizations
- **pytest**: Testing framework
- **great_expectations**: Advanced data validation (optional)

## License

MIT License

## Contact

**Shika Chaturvedi**
- GitHub: [@chaturvedi-shika](https://github.com/chaturvedi-shika)

## Roadmap

- [ ] Add Great Expectations integration
- [ ] Implement Z-score outlier detection
- [ ] Add ML-based anomaly detection
- [ ] Create PDF report export
- [ ] Add scheduled report generation
- [ ] Integration with data catalogs (AWS Glue, Databricks)
