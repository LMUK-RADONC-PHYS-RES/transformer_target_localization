# Transformer training strategies for real-time 2D target tracking in MRI-guided radiotherapy
Framework to train image registration Transformer networks with different unsupervised, supervised and 
patient-specific training strategies.

Elia Lombardo\
LMU Munich\
Elia.Lombardo@med.uni-muenchen.de

Evaluated with Python version 3.8.13 and PyTorch version 1.13.0

## Installation
* Download the repository to a local folder of your preference or clone the repository.
* Build a Docker image based on the provided `Dockerfile` and run a container 
while mounting the `transformer_target_localization` folder.
* Open `transformer_target_localization/code/config.py` and change `path_project` to the path inside the Docker 
container to the `transformer_target_localization` folder.
* If you do not want to use Weights and Biases to keep track of your trainings, delete everything related 
to `wandb` in the scripts. Otherwise, set your wandb API key in `transformer_target_localization/code/config.py` line 75.

## Usage
* In the file  `transformer_target_localization/code/config.py` you can set all the options for your
models, e.g. what type of training to perform and all hyper-parameter settings. 
* After that, run the corresponding main script in the terminal. For instance:
    * Set `model_name='TransMorph2D'` and `load_state=False` in `config.py`. Then execute `python main_train_unsup_TransMorph.py` 
    in the terminal to train TransMorph in an unsupervised fashion. 
    * Set `model_name='TransMorph2D'`, `load_state=True` and specify a pre-trained TransMorph with
    `start_time_string='name_of_pretrained_transmorph_results_folder'` in `config.py`. Then execute 
    `python main_train_sup_TransMorph.py` in the terminal to train TransMorph in an supervised fashion using the fine-tuning set.
    * Set `inference='validation'`, `model_name='TransMorph2D'`, `load_state=True`, and specify a pre-trained TransMorph with
    `start_time_string='name_of_pretrained_transmorph_results_folder'` in `config.py`. Then execute 
    `python main_train_ps_TransMorph.py` in the terminal to train TransMorph in an patient-specific fashion 
    using the first frames of each patient in the validation set.
    * Set `inference='testing'`, `model_name='TransMorph2D'`, `load_state=True`, and specify a pre-trained TransMorph with
    `start_time_string='name_of_pretrained_transmorph_results_folder'` in `config.py`. Then execute 
    `python main_train_ps_TransMorph.py` in the terminal to train TransMorph in an patient-specific fashion 
    using the first frames of each patient in the testing set.
* While the training scripts already perform inference (to be able to track validation performance),
proper inferece with statics is perfomed with the `main_infer` scripts:
    * Set `patient_specific_inference=False`, `inferece='validation'`, `model_name='TransMorph2D'` `load_state=True`, 
    and specify a trained TransMorph with `start_time_string='name_of_transmorph_results_folder'` in `config.py`.
    Then execute `python main_infer_TransMorph.py` in the terminal to run TransMorph on the validation set.
    * Set `patient_specific_inference=False`, `inferece='testing'`, `model_name='TransMorph2D'` `load_state=True`, 
    and specify a trained TransMorph with `start_time_string='name_of_transmorph_results_folder'` in `config.py`.
    Then execute `python main_infer_TransMorph.py` in the terminal to run TransMorph on the testing set. You can change the
    observer with the variable `observer_testing`.
    * Set `patient_specific_inference=False`, `inferece='validation'`, `model_name='TransMorph2D'` `load_state=True`, 
    and specify a trained TransMorph-Sup with `start_time_string='name_of_transmorph-sup_results_folder'` in `config.py`.
    Then execute `python main_infer_TransMorph.py` in the terminal to run TransMorph-Sup on the validation set (or testing set by changing `inference`). 
    * Set `patient_specific_inference=True`, `inferece='validation'`, `model_name='TransMorph2D'` `load_state=True`, 
    and specify trained TransMorph-PS models with `start_time_string='name_of_transmorph-ps_results_folder'` in `config.py`.
    Then execute `python main_infer_TransMorph.py` in the terminal to run TransMorph-PS models on the validation set (or testing set by changing `inference`). 
    * Set `inferece='validation'` and `model_name='Bspline'` in `config.py`. Then execute `python main_infer_Bspline.py` 
    to run Bspline on the dataset specified in `inference`. Change `Bspline` to `NoReg` or `InterObserver` to run these models instead.
* The weights of the models, metrics, plots, etc. will be saved under `transformer_target_localization/results`.

## Publication
If you use this code in a scientific publication, please cite our paper: 
https:xxx
