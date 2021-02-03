# Parapred - PyTorch

This repo is a PyTorch implementation of the original Parapred code from [Liberis
et al., 2018.](https://academic.oup.com/bioinformatics/article/34/17/2944/4972995). We would also like to point
users to the original [Github repo for Parapred](https://github.com/eliberis/parapred), written in Keras.

## Setup

We provide three ways to set up Parapred.

### Conda environment
```bash
conda env create -f environment.yml
make install
```

### Pip/requirements.txt
```bash
pip install -r requirements.txt
make install
```

### Dockerfile
Building the Docker image requires a bit of memory to download and install PyTorch in the Docker container.
We advise that you allocate at least 4GB of RAM to the Docker daemon.

For Mac, this can be set by `Docker > Preferences > Resources > Advanced`. 

```bash
docker build -t parapred_pytorch:latest 
```

## Running

To run Parapred, this depends on whether you used Conda/Pip or Docker.

### Conda/Pip
After install,
```bash
python cli.py predict <CDR_SEQUENCE> [OPTIONS]
```

e.g.
```bash
python cli.py predict CAKYPYYYGTSHWYFDVW -v -o output_ranibizumab.json
```

### Docker
The Docker equivalent is
```bash
docker run -v /tmp:/ --rm parapred_pytorch predict CAKYPYYYGTSHWYFDVW -v -o output_ranibizumab.json
```

## Additional notes

### Training
Parapred-pytorch is written in a very minimalist way to allow researchers to predict paratopes (CDR sequences
only) immediately. Currently, we only provide the pre-trained weights from the original Parapred publication
though there may be plans to include recipes for training in future releases.

The original Parapred method was based on the Chothia-defined CDRs based on the Chothia numbering. We provide
here a table mapping the CDR Chothia boundaries in the corresponding IMGT numbers. Note that these are not
identical to the IMGT boundaries of the CDRs.

| CDR | Chothia numbers | IMGT numbers | 
| --- | --------------- | ------------ |
| H1  |  H26-H34        | H27-H37 |
| H2  |  H52-H56        | H57-H64 |
| H3  |  H95-H102       | H107-H117 |
| L1  |  L24-L34        | L24-40 |
| L2  |  L50-L56        | L56-L69 |
| L3  |  L89-L97 | L105-L117|

### LSTM activations
Our implementation provides users the ability to test Parapred using `sigmoid` activations for the
LSTM layer. The original Parapred method was based on Tensorflow 1.2/Keras, which had used `hard_sigmoid`
activations in the LSTM step. We retain this hard sigmoid as the default. 