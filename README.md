

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

We recommend using [Conda](https://conda.io/en/latest/index.html) to install our package. An anaconda environment can be set up with the following command:
```
conda env create -f env.yml
```
Systems requirements: Linux / Windows (64 bit)
### A quick example
The following codes could be used to prepare training dataset and train the Riboformer model to predict the ribosome profiles in E. coli with the high Mg/flash frozen protocol (Mohammad et al., 2019). The source data is retrieved from [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119104). The genome sequence and gene positions are retrieved from genome build NC_000913.2.
```
python data_processing.py
python training.py -e 15 -l 0.0005 --save
```
Our algorithm can be run on either CPU or GPU (requires cuda). The running time is ~15 min on a V100 GPU (16GB). Results will be saved in the ```/models``` folder.


### Training Riboformer on new dataset
The following files are required for generating training dataset for Riboformer:
- ```inputdata_f(r).wig```,  which store ribosome coverage data for the reference ribosome profiling dataset. Each line specifies a position in the genome and the number of ribosome footprints (or reads) at that position. Two files representing the forward (f) and reverse (r) direction should be provided:
	```
   track type=wiggle_0 name=tracklabel viewLimits=-5:5 color=83,8,86
   fixedStep  chrom=NC_000913.2  start=1  step=1
   0.0
   0.0
   0.0
	```
	Please note that the ```chrom``` name should match the chromosome names in the genome sequence and genome annotations. For example, the wig file shown here is using the reference genome of *E. coli* MG1655 ([NC_000913.2](https://www.ncbi.nlm.nih.gov/nuccore/49175990)).
-  ```outputdata_f(r).wig```,  which store ribosome coverage data for the target ribosome profiling dataset. Each line specifies a position in the genome and the number of ribosome footprints (or reads) at that position. Two files representing the forward (f) and reverse (r) direction should be provided.
- ```genome_sequence.fasta```, which stores the genomic sequence for the organism.
- ```gene_annotation.gff3```, which stores gene annotations for the organism.

The genome sequences and gene annotation files could be directly downloaded from NCBI. Sample data for prokaryotic cells are included in ```/datasets/GSE119104_Mg_buffer/``` . Sample data for eukaryotic cells are included in ```/datasets/GSE139036_disome/``` . All the files should be placed in one data folder. The training dataset could be prepared using ```data_processing.py``` and the output files will be used to train the Riboformer model.

To run the function, execute the following command:
```
python data_processing.py [-h] [-w WSIZE] [-d DATA_DIR] [-r REFERENCE] [-t TARGET] [-th THRESHOLD] [-p PSITE]
```
The function accepts the following optional arguments:

- `-h, --help`: Show the help message and exit.
- `-w WSIZE, --wsize WSIZE`: Set the window size for model training (default: 40).
- `-d DATA_DIR, --data_dir DATA_DIR`: Set the data folder name in `/datasets/`. (default: 'GSE119104_Mg_buffer').
- `-r REFERENCE, --reference REFERENCE`: Set the name of reference dataset. (default: 'GSM3358138_filter_Cm_ctrl', the corresponding file names are 'GSM3358138_filter_Cm_ctrl_f.wig' and 'GSM3358138_filter_Cm_ctrl_r.wig' ).
- `-t TARGET, --target TARGET`: Set the name of target dataset. (default: 'GSM3358140_freeze_Mg_ctrl', the corresponding file names are 'GSM3358140_freeze_Mg_ctrl_f.wig' and 'GSM3358140_freeze_Mg_ctrl_r.wig').
- `-p PSITE, --psite PSITE`: We applied uniform offsetting in data preprocessing and this parameter defines the offset from the ends of the aligned fragments (default: 14, Mohammad et al., eLife 2019).
- `-th THRESHOLD, --threshold THRESHOLD`: For the efficient analysis of data, our algorithm includes the top quartile of genes based on Ribosome Density (RD). (default: 25, include the top 25% genes).

The function automatically loads gene annotations (ending in `.gff3`) and genome sequences (ending in `.fasta`) from the data folder. The output will be three files in the same data folder, storing the input and output data for all codons of interests and the genome positions for each codon. These datasets can then be used for model training.


### Applying trained Riboformer model on new dataset
Process the new dataset using ```data_processing.py``` and put it in the data folder. To apply a trained Riboformer model, execute the following command:
```
python transfer.py [-h] [-i INPUT_FOLDER] [-m MODEL_FOLDER]
```
The script accepts the following optional arguments:

- `-h, --help`: Show the help message and exit.
- `-i INPUT_FOLDER, --input_folder INPUT_FOLDER`: Set the input data folder. This folder should contain the processed dataset.
- `-m MODEL_FOLDER, --model_folder MODEL_FOLDER`: Set the model folder. This folder should contain a trained Riboformer model.

We have provided sample data for this function: 
```
cd Riboformer/Riboformer
python transfer.py -i ../datasets/GSE139036_disome -m ../models/yeast_disome
```
This will produce a file named `model_prediction.txt `storing the predicted ribosome densities in the input folder. The corresponding codon positions are stored in `zc.txt`
 file in the input datasets.

### Pretrained models
We provide 5 pretrained models that could be used to reproduce results in our work. The list of available pretrained models:
| Model name | Training dataset | Description |
|----------|----------|----------|
| bacteria_cm_mg |  Mohammad et al., eLife 2019 | predict ribosome profile with high Mg/flash frozen protocol in E. coli |
| yeast_disome | Meydan & Guydosh, Mol Cell 2020 | predict disome profile in yeast |
| yeast_aging | Stein et al., Nature 2022 | predict ribosome profile in aged yeast (day 4) |
| worm_aging | Stein et al., Nature 2022 | predict ribosome profile in aged worm (day 12) |
| covid_model | Finkel et al., Nature 2021 | predict ribosome profile for SARS-CoV-2 (24 hpi) |

### Comparison of Riboformer with baseline methods
We implemented RiboMIMO and Riboexp based on the source code provided from the original research (https://github.com/tiantz17/RiboMIMO, and https://github.com/Liuxg16/Riboexp). To fairly compare Riboexp and RiboMIMO with our model, we truncated the reference input branch in Riboformer to make it a purely sequence-based model (seq-only mode). All models were trained on the same set of highly expressed genes using cross-validation tests. The training datasets for the two models are available in the `benchmarking` folder.

### Reproducibility
 `reproducibility/riboformer_artifact_correct.ipynb` :  reproduce Fig 1f, 1g, Supp fig 2, 3.

> Related dataset can be downloaded at [Google drive](https://drive.google.com/file/d/1B5RV_74uPLYjpakOdUmH03_NMP0hQXrB/view?usp=sharing).

`reproducibility/riboformer_yeast_SIS.ipynb` :  reproduce Fig 2b, 2c, 2e.

> Related dataset can be downloaded at [Google drive](https://drive.google.com/file/d/1F8mwXFDC9ufXTsuWQEP6g_PGHD23cGjV/view?usp=sharing).


Model prediction and source data for GSE77617, GSE152664, GSE152850 and GSE165592 can be assessed at [Google drive](https://drive.google.com/file/d/1XXmyePpJDK5RkbrF1tRkVu8EFFiy-rOd/view?usp=sharing)

### Reference
- [Riboformer: A Deep Learning Framework for Predicting Context-Dependent Translation Dynamics](https://www.biorxiv.org/content/10.1101/2023.04.24.538053v1)
- Please contact shaobin@broadinstitute.org or raise an issue in the github repo with any questions about installation or usage.
