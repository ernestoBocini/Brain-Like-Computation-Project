## Brain-Like-Computation-Project
### Group project for NX-414 course at EPFL 

### Authors (team: Team_BAK)
- Ernesto Bocini: [@ernestoBocini](https://github.com/ernestoBocini)
- Florence Crozat [@florence11](https://github.com/florence11)

## File structure

#### Report
- report.pdf: one page report of the project including explanations of methods and our results.

#### PY files:
- test.py
  - File that contains the best performing model. Use this file to reproduce our results or to test your dataset.
- utils.py
  - File that contains various helper functions for the project.

#### Notebooks:
- EDA.ipynb:
  - file containing exploratory data analysis
- week5.ipynb:
  - contains ridge regression, ridge regression with PCA and resnet50
- week6.ipynb:
  - contains shallow CNN with optimization
- week7.ipynb
    - contains best model and cornet models
    
#### Data folder:
- activationsResNet50: activations for week5
- cornet-useful: cornet weights and images. See also [cornet models](https://github.com/dicarlolab/CORnet)
- resnet50_improved: result submission


## How to reproduce our results
We assume that the repository is already downloaded and extracted, that the [IT_data.h5](https://drive.google.com/file/d/1s6caFNRpyR9m7ZM6XEv_e8mcXT3_PnHS/view?usp=share_link) is downloaded and extracted in the data folder at the root of the program. We further assume that Anaconda is already installed.

### Create the environment
Make sure your environment satisfies the following fundamental requirements:
- Python 3.7+
- NumPy module 
- PyTorch 1.13 module
- matplotlib module

### Required packages
- Required packages for the best model:
  - h5py
  - os
  - Image from PIL
  - pickle
  - resnet50, ResNet50_Weights from torchvision.models
  - tqdm
  - explained_variance_score from sklearn.metrics
- Required packages for running all the notebooks:
  - all packages above
  - optuna


### Run the code
From the root folder of the project

```shell
python test.py
```
Careful: training might be time consuming. The model has been trained and runned using the following machine:
   - 16 vCPU, 104 GB di RAM, NVIDIA T4 x 1 .
