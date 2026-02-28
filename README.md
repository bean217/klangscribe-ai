# klangscribe-ai
Repository for KlangScribe Model Training &amp; Inference.

## Dataset Preprocessing

If using the [KlangScribe Official Dataset](https://drive.google.com/drive/folders/1akCO8kormDrm5N30WHDyIISBr74YWtxT?usp=sharing), the [Dataset Preprocessing](/dataset_preprocessing) module contains functionality for producing a version of the dataset that KlangScribe can train on directly. Please refer to [it's documentation](/dataset_preprocessing/README.md) for more information on how this is done.

This should be used if you are looking to retrain KlangScribe using a different data modeling configuration. By default, KlangScribe uses the following data model configuration (also located [here](/configs/defaults/data_model.yaml)):
```
TBD; not yet defined
```


## Training

To train KlangScribe, instead of using an existing model version, following these steps:

1) Preprocess your dataset by following the steps in [Dataset Preprocessing](#dataset-preprocessing)

2) TBD