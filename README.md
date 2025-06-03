Here is the cleaned and corrected version of your GitHub README markdown:

```markdown
# DemandForecast

This repository contains a synthetic dataset (`data/OrangeJuiceX25.csv`) and prototype code for demand forecasting models.

## Dataset

The file `data/OrangeJuiceX25.csv` simulates weekly sales of multiple orange juice brands across several stores. Each row includes the store, brand, week number, eleven price variables, two promotion flags (`deal` and `feat`), and the `sales` target.

**Example rows:**

```

store,brand,week,price1,price2,price3,price4,price5,price6,price7,price8,price9,price10,price11,deal,feat,sales
1,1,1,3.04,1.04,4.52,1.34,3.35,1.99,1.62,2.6,1.2,1.96,2.67,0,0,109.2
1,1,2,3.83,3.1,3.26,2.78,2.26,3.28,2.12,2.15,3.3,2.97,4.17,1,1,139.4

````

## Data Ingestion and Preprocessing

The script `src/data_ingestion_preprocessing.py` demonstrates how to:

1. Load the dataset with `pandas`
2. Explore summary statistics
3. Visualize sales distribution and price correlations
4. Engineer features such as lagged sales and seasonal indicators

Run the script with:

```bash
python src/data_ingestion_preprocessing.py
````

This will print basic dataset information and display a sample of the processed DataFrame.

## Econometric Model Prototypes

The script `src/econometric_models.py` includes prototype implementations for:

* **LASSO Regression per SKU×Store** using rolling windows and `LassoCV`
* **General-to-Specific (GETS) OLS** with t-ratio pruning
* **Panel-Wide LASSO** pooling all store–brand data

Run the models and compute RMSE:

```bash
python src/econometric_models.py
```

**Dependencies:**

* `pandas`
* `scikit-learn`
* `statsmodels`

## XGBoost Prototypes

The script `src/xgboost_models.py` implements gradient boosting models using `xgboost`:

* **Per-series XGBoost**: tuned per store–brand pair using rolling windows
* **Pooled XGBoost**: stacks all series and adds store/brand dummy variables

To run both variants:

```bash
python src/xgboost_models.py
```

To run only the pooled version:

```bash
python src/pooled_xgboost.py
```

**Dependencies:**

* `xgboost`
* `pandas`
* `scikit-learn`

## Transformer Prototype

The script `src/transformer_trt.py` provides a small Transformer model inspired by the Temporal Fusion Transformer for next-week sales prediction. It includes store and brand embeddings and a couple of self-attention layers.

Run with:

```bash
python src/transformer_trt.py
```

**Dependencies:**

* `pytorch`
* All dependencies listed under Econometric and XGBoost prototypes

## Sequence Models

The script `src/sequence_models.py` implements:

* A per-series LSTM
* A simple Seq2Seq model

Run with:

```bash
python src/sequence_models.py
```

**Dependencies:**

* `pytorch`
* `scikit-learn`

---

**Note:** Ensure all required packages are installed (e.g., via `pip install -r requirements.txt`, if available). Adjust file paths as needed.

```

### Fixes applied:

- Removed unnecessary backtick fencing inside code blocks.
- Corrected small formatting inconsistencies (e.g., bullet symbols, extra closing backticks).
- Harmonized phrasing and made sections more parallel in tone and formatting.
- Cleaned note and command syntax for clarity.
```
