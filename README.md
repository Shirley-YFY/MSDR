# MSDR

MSDR: Multi-Step Dependency Relation Networks for Spatial Temporal Forecasting



## GMSDR For Task Of Demand Prediction

```shell
# Taxi
python main.py --type=taxi

# Bike
python main.py --type=bike
```



## GMSDR For Task Of Speed Prediction

### Data Preparation

```shell
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```



### Model Training

```shell
# METR-LA
python main.py --config_filename=data/model/GMSDR_LA.yaml

# PEMS-BAY
python main.py --config_filename=data/model/GMSDR_BAY.yaml
```



