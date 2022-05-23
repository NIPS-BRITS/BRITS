# Bidirectional Recurrent Imputation for Time Series

Time series are widely used as signals in many classification/regression tasks. It is ubiquitous that time series contains many missing values. Given multiple correlated time series data, how to fill in missing values and to predict their class labels? Existing imputation methods often impose strong assumptions of the underlying data generating process, such as linear dynamics in the state space. In this paper, we propose BRITS, a novel method based on recurrent neural networks for missing value imputation in time series data. Our proposed method directly learns the missing values in a bidirectional recurrent dynamical system, without any specific assumption. The imputed values are treated as variables of RNN graph and can be effectively updated during the backpropagation.BRITS has three advantages: (a) it can handle multiple correlated missing values in time series; (b) it generalizes to time series with nonlinear dynamics underlying; (c) it provides a data-driven imputation procedure and applies to general settings with missing data.We evaluate our model on three real-world datasets, including an air quality dataset, a health-care data, and a localization data for human activity. Experiments show that our model outperforms the state-of-the-art methods in both imputation and classification/regression accuracies.


This repo contains the code for the BRITS paper:

```
Wei Cao, Dong Wang, Jian Li, Hao Zhou, Yitan Li and Lei Li, "BRITS: Bidirectional Recurrent Imputation for Time Series", In Neural Information Processing Systems (NeurIPS), 2018.
```

If you use the code, please cite using the following bib entry:

@InProceedings{cao2018brits,
  author    = {Wei Cao and Dong Wang and Jian Li and Hao Zhou and Yitan Li and Lei Li},
  title     = {BRITS: Bidirectional Recurrent Imputation for Time Series},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1805.10572},
}

# Description
The source codes of RITS-I, RITS, BRITS-I, BRITS for health-care data imputation/classification

To run the code:

python main.py --epochs 1000 --batch_size 32 --model brist

# Data Format
In json folder, we provide the sample data (400 patients).
The data format is as follows:

* Each line in json/json is a string represents a python dict
* The structure of each dict is
    * forward
    * backward
    * label

    'forward' and 'backward' is a list of python dicts, which represents the input sequence in forward/backward directions. As an example for forward direction, each dict in the sequence contains:
    * values: list, indicating x_t \in R^d (after elimination)
    * masks: list, indicating m_t \in R^d
    * deltas: list, indicating \delta_t \in R^d
    * forwards: list, the forward imputation, only used in GRU_D, can be any numbers in our model
    * evals: list, indicating x_t \in R^d (before elimination)
    * eval_masks: list, indicating whether each value is an imputation ground-truth

# Data Download Links

* Air Quality Data:
URL: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip

* Health-care Data:
URL: https://physionet.org/challenge/2012/
We use the test-a.zip in our experiment.

* Human Activity Data:
URL: https://archive.ics.uci.edu/ml/datasets/Localization+Data+for+Person+Activity
