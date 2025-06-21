# Natural Cycles Assignment - Pregnancy Analysis

Jeroen Buil's submission for the coding assessment for the Senior Data Scientist position at Natural Cycles, focusing on pregnancy and conception time analysis using machine learning approaches.

## 📋 Project Overview

This project analyzes pregnancy data to answer key questions about conception time and factors affecting pregnancy success. The analysis includes, explorative data analysis, and both statistical and machine learning approaches to understand patterns in fertility data.

## 🎯 Analysis Questions

The project addresses four main research questions:

1. **Pregnancy Chance Analysis**: What is the chance of getting pregnant within 13 cycles?
2. **Conception Time Distribution**: What is the usual time it takes to get pregnant?
3. **Factor Impact Analysis**: What factors impact the time it takes to get pregnant?
4. **Machine Learning Approaches**: How would your approach change if you were to use different techniques (e.g., ML or non-ML
methods)? What trade-offs would you consider?
   - Classification: Factors impacting conception time (binary classification)
   - Regression: Factors impacting conception time (regression analysis)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd natural-cycles-assignment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```


## 📊 Data

**⚠️ IMPORTANT**: All analyses in this project rely on the Natural Cycles dataset (`ncdatachallenge-2021-v1.csv`) which is **NOT INCLUDED** in this repository.

### Required Data Setup

To run any analysis, you must:

1. **Obtain the dataset** from Natural Cycles
2. **Place the file** in the `data/external/` directory
3. **Ensure the filename** is exactly `ncdatachallenge-2021-v1.csv`

The dataset contains:
- Pregnancy outcomes
- Cycles trying to conceive
- Various demographic and health factors
- Lifestyle and medical information

**Note**: Without this data file, all analysis scripts will fail with a "file not found" error.

## 🔬 Analysis Modules

### Core Analysis Scripts

- **`question_1_pregnancy_chance_13_cycles.py`**: Analyzes pregnancy success rates within 13 cycles
- **`question_2_usual_conception_time.py`**: Examines the distribution of conception times
- **`question_3_factors_impacting_conception_time.py`**: Statistical analysis of factors affecting conception
- **`question_4_ML_classification_approach_factors_impacting_conception_time.py`**: Machine learning classification approach (good performance)
- **`question_4_ML_regression_approach_factors_impacting_conception_time.py`**: Machine learning regression approach (poor performance)

### Utility Modules

- **`import_data.py`**: Data loading and cleaning utilities
- **`plotting.py`**: Visualization functions with consistent styling
- **`model.py`**: Machine learning model training and evaluation
- **`utils.py`**: Helper functions for data preprocessing
- **`config.py`**: Configuration settings

## 🎨 Visualization Features

- **Consistent Styling**: All plots use a unified visual style with viridis colormap
- **High-Quality Outputs**: Figures saved at 200 DPI for optimal quality
- **Comprehensive Plots**: 
  - Feature importance visualizations
  - Predicted vs actual plots
  - Correlation matrices
  - Distribution plots
  - Model comparison charts

## 📈 Key Findings

The analysis provides insights into:
- Pregnancy success rates and timing
- Factors most strongly associated with conception time
- Machine learning models for predicting conception outcomes
- Statistical relationships between variables

## 🛠️ Usage Examples

### Run Individual Analysis

```bash
# Pregnancy chance analysis
python -m natural_cycles_assignment.question_1_pregnancy_chance_13_cycles

# Conception time analysis
python -m natural_cycles_assignment.question_2_usual_conception_time

# Factor impact analysis
python -m natural_cycles_assignment.question_3_factors_impacting_conception_time

# Machine learning classification
python -m natural_cycles_assignment.question_4_ML_classification_approach_factors_impacting_conception_time

# Machine learning regression
python -m natural_cycles_assignment.question_4_ML_regression_approach_factors_impacting_conception_time
```

### 📊 Generated Outputs

**⚠️ Important**: Running each analysis will automatically generate figures that are stored in the `reports/figures/` folder. These include:
- Pregnancy chance visualizations
- Conception time distribution plots
- Factor impact analysis charts
- Machine learning model performance plots
- Feature importance visualizations

**📋 Comprehensive Report**: A PowerPoint presentation (`Natural Cycles - Assignment Report - Jeroen Buil.pptx`) is included in the `reports/` folder that provides:
- Complete overview of all analysis results
- Context and interpretation of findings
- Key insights and conclusions
- Recommendations based on the analysis

## 📁 Project Structure

```
natural-cycles-assignment/
├── data/                          # Data files
│   └── external/                  # External datasets
├── docs/                          # Documentation
├── models/                        # Trained models
├── natural_cycles_assignment/     # Source code
│   ├── modeling/                  # ML model code
│   ├── *.py                      # Analysis scripts
│   └── ...
├── notebooks/                     # Jupyter notebooks
├── reports/                       # Generated reports
│   ├── figures/                   # Generated plots (auto-created)
│   └── Natural Cycles - Assignment Report - Jeroen Buil.pptx  # Complete results overview
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## 🔧 Development

### Code Quality

The project uses:
- **Ruff**: For linting and code formatting
- **Type hints**: For better code documentation
- **Modular design**: For maintainable and testable code

### Adding New Analysis

1. Create a new script in `natural_cycles_assignment/`
2. Use the existing utility functions from `plotting.py`, `model.py`, etc.
3. Follow the established naming conventions
4. Update this README if adding new functionality

## 📊 Outputs

All analyses generate:
- **Console output**: Statistical summaries and model performance metrics
- **Visualizations**: Saved as PNG files in `reports/figures/`
- **Model artifacts**: Trained models and evaluation results

## 🤝 Contributing

1. Follow the existing code style and structure
2. Add appropriate documentation
3. Ensure all tests pass
4. Update requirements if adding new dependencies

## 📄 License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Jeroen Buil** - Senior Data Scientist

---

*This project demonstrates advanced data science techniques applied to fertility and pregnancy data analysis, showcasing both statistical and machine learning approaches.*

