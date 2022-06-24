# Top_Tox
It is a module of multiple classification models that predicts the hepatotoxicity and classifies the compounds as "Toxic" or "Non-Toxic"

#### environment.yml file currently works only under Linux. If you want to use this code, Install pip requirement from requirements.txt


## Installation with Anaconda

### Clone the reopsitory to your desired directory
```
git clone https://github.com/Datadriven1/Top_Tox.git
```
```
cd Top_Tox
```
#### Create new conda environment with Python 3.8.13
```
conda env create -f environment.yml
```
### Activate the environment
```
conda activate top_tox
```
### Install pip dependencies
```
pip install -r requirements.txt
```
### Training all classification model
```
python training.py --input_file "Your file name"
```
## Citation
#### If you use this code or data, please cite:

Top Tox method paper:
