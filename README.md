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
python3 data_processing.py
python3 training.py -e=15 -l=0.0005 --save
```


### Running Riboformer for new dataset
The following files are required for generating training dataset for Riboformer:
- ```ribosome_density_f(r).wig```,  which store ribosome coverage data. Each line specifies a position in the genome and a signal value, usually representing the number of ribosome footprints (or reads) at that position. Two files representing the forward (f) and reverse (r) direction should be provided for the reference dataset and the target dataset.
- ```genome_sequence.fasta```, which stores the genomic sequence for the organism.
- ```gene_positions.csv```, which stores positions of all the genes. Each line specifies the starting position, ending position, and the direction for one gene (1 for forward and 2 for reverse).

All the files should be placed in one folder. The training dataset could be prepared using ```data_processing.py``` and the output files will be used to train the Riboformer model.

### Pretrained models
We provide 5 pretrained models that could be used to reproduce results in our work. The list of avaiable pretrained models:
| Model name | Training dataset | Description |
|----------|----------|----------|
| bacteria_cm_mg |  Mohammad et al., eLife 2019 | predict ribosome profile with high Mg/flash frozen protocol in E. coli |
| yeast_disome | Meydan & Guydosh, Mol Cell 2020 | predict disome profile in yeast |
| yeast_aging | Stein et al., Nature 2022 | predict ribosome profile in aged yeast (day 4) |
| worm_aging | Stein et al., Nature 2022 | predict ribosome profile in aged worm (day 12) |
| covid_model | Finkel et al., Nature 2021 | predict ribosome profile for SARS-CoV-2 (24 hpi) |

### Contact
Please contact shaobin@broadinstitute.org or raise an issue in the github repo with any questions about installation or usage.
