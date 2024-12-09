# Bachelor-Thesis

## model-training
The interplay of the code files in model-training looks as follows: most of these files rely on each other, however there is one important distinction - the training and the testing are separate. So during training, which will yield the training metrics and the validation metrics of the latest, best and final model, no evaluation on the test set takes place. The files working together to guarantee a smooth training procedure are:
- mm-pt_config.yaml
- main.py
- data.py
- multi_head_multi_domain_pt.py
- backbones.py
- model.py
- utils.py
For the training, multi_head_multi_domain_pt.py contains by far the most logic. This includes the sampling procedure of the multi-domain multi-task training paradigm of Woerner et al. (2024)

The testing then yields the test metrics as well as a csv files with the predictions for the test samples. The files working together to guarantee a smooth testing procedure are:
- mm-pt_config.yaml
- test.py
- data.py
- backbones.py
- multi_head_multi_domain_pt.py
- model.py
- utils.py

## reevaluation
This part of the code is focused on the reevaluation of the models presented by Doerrich et al. (2024). The existing models were reevaluated to obtain the metrics used in the bespoke benchmark presented in this thesis. Some parameters like kNN_true in evaluate.py thus shouldn't be viewed as static with a fixed value.

Brief summary of the functionality of the files:
evaluate.py calculates the test metrics for a model trained on an individual dataset
reevaluate.py takes csv files of the individual models and stores their performance for all resolutions separately in a csv file dedicated to the dataset the model was trained on
reevaluation-aggregate.py calculates the average metric for a model based on the model's performance on all the datasets for each resolution separately

Both the separation of training and testing as well as the hardcoded use of parameters are not necessary for the code to run and thus could be improved by rewriting it, including testing immediately after training such that in order to fully train and test a model, only one command, namely running main.py would need to be executed. Hardcoded variables could be handled more elegantly by offering the possibility to specify them in e.g. a .yaml file or iterating over all the possible values and doing the reevaluation for all possible combinations sequentially.



Note: paths and access tokens were replaced with placeholders where possible and remembered.


Cited papers:
Woerner, S., Jaques, A., & Baumgartner, C.F. (2024). A comprehensive and easy-to-use multi-domain multi-task medical imaging meta-dataset (MedIMeta). ArXiv, abs/2404.16000.
Doerrich, S., Di Salvo, F., Brockmann, J., & Ledig, C. (2024). Rethinking Model Prototyping through the MedMNIST+ Dataset Collection. ArXiv, abs/2404.15786.

## Web-Development

