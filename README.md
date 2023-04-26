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
- EDA.ipynb
- week5.ipynb
- week6.ipynb
    - File containing all 6 implementations of ML functions required for the project.
- week7.ipynb
    - File containing functions to load and preprocess the data.
    
#### Data folder:
- activationsResNet50: train data needs to be uploaded and placed here with this name
- cornet-useful: test data needs to be uploaded and placed here with this name
- resnet50_improved: result submission


## How to reproduce our results
We assume that the repository is already downloaded and extracted, that the [IT_data.h5](https://drive.google.com/file/d/1s6caFNRpyR9m7ZM6XEv_e8mcXT3_PnHS/view?usp=share_link) is downloaded and extracted in the data folder at the root of the program. We further assume that Anaconda is already installed.

### Create the environment
Make sure your environment satisfies the following requirements:
- Python 3.7+
- NumPy module 
- matplotlib

### Run the code
From the root folder of the project

```shell
python test.py
```
