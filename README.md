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