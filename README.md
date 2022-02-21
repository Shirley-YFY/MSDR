# MSDR

MSDR: Multi-Step Dependency Relation Networks for Spatial Temporal Forecasting



## GMSDR For Task Of Demand Prediction
The traffic data files, i.e., taxi_data.h5 and bike_data.h5, are available at [Baidu Yun](https://pan.baidu.com/s/1mFeS8-WcbzndPXGaJDGDEQ?pwd=tstw), and should be put into the `data/nogrid/` folder. 
```shell
# NYC Taxi
python main.py --type=taxi

# NYC City Bike
python main.py --type=bike
```



## GMSDR For Task Of Speed Prediction

### Data Preparation
The traffic data files, i.e., metr-la.h5 and pems-bay.h5, are available at [Baidu Yun](https://pan.baidu.com/s/1mFeS8-WcbzndPXGaJDGDEQ?pwd=tstw), and should be put into the data/ folder. 
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



