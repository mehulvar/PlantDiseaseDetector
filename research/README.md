# Research (original exploration)

The original scripts from the first version of this project, kept for reference. Run them from this
directory (`cd research`). They read `../data/plant_health_data.csv` and `../Image Data/`.

- `NumericalData.py` - exploratory data analysis of the soil/sensor dataset
- `InitialSVMForNumericalData.py`, `RandomForest.py`, `GradientBoosting.py` - first soil classifiers
- `CNN.py` - first image model (custom Keras CNN, 13 classes, 88% test accuracy); its outputs are in
  `legacy_keras_model/` and `figures/`
- `createMetadata.py`, `CountImages.py` - dataset utilities
- `RotatingImages.py` - augmentation helper carried over from an earlier project (not used here)

The production pipeline that replaced these lives in `../ml` (training) and `../backend` (serving).
