# Riboformer: a Deep Learning Framework for Predicting Context-Dependent Translation Dynamics

![title](https://user-images.githubusercontent.com/12596418/232886009-400f779b-b23d-489c-b52f-194da79a4e5c.png)

Translation elongation is essential for maintaining cellular proteostasis and changes in the translational landscape are associated with age-related decline in protein homeostasis. We developed Riboformer, a deep learning-based framework for predicting context-dependent changes in translation dynamics using a transformer architecture. Riboformer accurately predicts ribosome densities at the codon level, corrects experimental artifacts, and identifies sequence determinants underlying ribosome collision across various biological contexts. Our tool offers a comprehensive and interpretable approach for standardizing ribosome profiling datasets and facilitating biological discoveries.

### Installation
Python package dependencies:
- tensorflow
- keras
- seaborn
- pandas
- juypterlab

An anaconda environment can be set up with the following command:
```
conda env create -f env.yml
```

### A quick example
The following codes could be used to prepare training dataset and train the Riboformer model to predict the ribosome profiles in E. coli with the high Mg/flash frozen protocol (Mohammad et al., 2019). The source data is retrieved from [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119104). The genome sequence and gene positions are retrieved from genome build NC_000913.2.
```
python data_processing.py
python training.py -e=15 -l=0.0005 --save
```
The running time is ~15 min on a V100 GPU (16GB). Results will be saved in the ```/models``` folder.


### Training Riboformer on new dataset
The following files are required for generating training dataset for Riboformer:
- ```ribosome_density_f(r).wig```,  which store ribosome coverage data. Each line specifies a position in the genome and a signal value, usually representing the number of ribosome footprints (or reads) at that position. Two files representing the forward (f) and reverse (r) direction should be provided for the reference dataset and the target dataset. Values are tab-separated.
- ```genome_sequence.fasta```, which stores the genomic sequence for the organism.
- ```gene_positions.csv```, which stores positions of all the genes. Each line specifies the starting position, ending position, and the direction for one gene (1 for forward and 2 for reverse). Values are tab-separated.

All the files should be placed in one data folder. The training dataset could be prepared using ```data_processing.py``` and the output files will be used to train the Riboformer model.

To run the function, execute the following command:
```
python data_processing.py [-h] [-w WSIZE] [-d DATA_DIR] [-r REFERENCE] [-t TARGET]
```
The function accepts the following optional arguments:

- `-h, --help`: Show the help message and exit.
- `-w WSIZE, --wsize WSIZE`: Set the window size for model training (default: 40).
- `-d DATA_DIR, --data_dir DATA_DIR`: Set the data folder name (default: '/datasets/GSE119104_Mg_buffer/').
- `-r REFERENCE, --reference REFERENCE`: Set the reference dataset name (default: 'GSM3358138_filter_Cm_ctrl').
- `-t TARGET, --target TARGET`: Set the target dataset name (default: 'GSM3358140_freeze_Mg_ctrl').

The function automatically loads gene positions and genome sequences from the data folder.


### Applying trained Riboformer model on new dataset
Process the new dataset using ```data_processing.py``` and put it in the data folder. To apply a trained Riboformer model, execute the following command:
```
python transfer.py [-h] [-i INPUT_FOLDER] [-m MODEL_FOLDER]
```
The script accepts the following optional arguments:

- `-h, --help`: Show the help message and exit.
- `-i INPUT_FOLDER, --input_folder INPUT_FOLDER`: Set the input data folder.
- `-m MODEL_FOLDER, --model_folder MODEL_FOLDER`: Set the model folder.


### Pretrained models
We provide 5 pretrained models that could be used to reproduce results in our work. The list of available pretrained models:
| Model name | Training dataset | Description |
|----------|----------|----------|
| bacteria_cm_mg |  Mohammad et al., eLife 2019 | predict ribosome profile with high Mg/flash frozen protocol in E. coli |
| yeast_disome | Meydan & Guydosh, Mol Cell 2020 | predict disome profile in yeast |
| yeast_aging | Stein et al., Nature 2022 | predict ribosome profile in aged yeast (day 4) |
| worm_aging | Stein et al., Nature 2022 | predict ribosome profile in aged worm (day 12) |
| covid_model | Finkel et al., Nature 2021 | predict ribosome profile for SARS-CoV-2 (24 hpi) |

### Reference
- [Riboformer: A Deep Learning Framework for Predicting Context-Dependent Translation Dynamics](https://www.biorxiv.org/content/10.1101/2023.04.24.538053v1)
- Please contact shaobin@broadinstitute.org or raise an issue in the github repo with any questions about installation or usage.
