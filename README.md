# DemandForecast

This repository contains a synthetic dataset `OrangeJuiceX25.csv` and prototype code for demand forecasting models.

## Dataset

`data/OrangeJuiceX25.csv` simulates weekly sales of multiple orange juice brands across several stores. Each row records the store, brand, week number, eleven price variables, two promotion flags (`deal` and `feat`), and the `sales` target.

Example rows:

```
store,brand,week,price1,price2,price3,price4,price5,price6,price7,price8,price9,price10,price11,deal,feat,sales
1,1,1,3.04,1.04,4.52,1.34,3.35,1.99,1.62,2.6,1.2,1.96,2.67,0,0,109.2
1,1,2,3.83,3.1,3.26,2.78,2.26,3.28,2.12,2.15,3.3,2.97,4.17,1,1,139.4
```

## Data Ingestion and Preprocessing

The script `src/data_ingestion_preprocessing.py` demonstrates how to load the dataset with `pandas`, explore summary statistics, visualize sales distribution and price correlations, and engineer features such as lagged sales and seasonal indicators.

Run it with:

```bash
python src/data_ingestion_preprocessing.py
```

This will print basic dataset information and show a sample of the processed DataFrame.