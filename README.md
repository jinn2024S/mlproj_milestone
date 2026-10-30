## Getting the data
1. Go to the HAM10000 dataset on Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
2. Download Original FormatZip file (3.0GB)
3. Unzip the image zips and put everything in the project folder, so it looks like this:
```
project-folder/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
├── HAM10000_metadata
├── ISIC2018_Task3_Test_Images/
├── ISIC2018_Task3_Test_GroundTruth.csv
├── prepare_data.py
├── lesion_dataset.py
├── train.ipynb
└── evaluate_test.ipynb
```

## Running the project
1. Install the packages:
```
   pip install -r requirements.txt
```
2. Create the labels and the train/validation/test split:
```
   python prepare_data.py
```
3. Train three models: open `train.ipynb`, set `SEED = 1`, and run all cells. Repeat with `SEED = 2` and `SEED = 3`, restarting the kernel each time.
4. Evaluate all three models on the test set: run all cells in `evaluate_test.ipynb`.
