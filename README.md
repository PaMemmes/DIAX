# Increasing Detection Rate for Imbalanced Malicious Traffic using Generative Adversarial Networks
This repository includes the code to the paper "Increasing Detection Rate for Imbalanced Malicious Traffic using Generative Adversarial Networks" of EICC 2024 (Note: The "DIAX" model here is called "combined".).
It includes 4 models: A GAN, WGAN, XGBoost and the DIAX model.

First run
``` bash
conda env create -f environment.yml
```
to get all needed dependencies for the python code.

## Models
Please also download the [cicids2018 data set](https://registry.opendata.aws/cse-cic-ids2018/), this needs to be put in path: /mnt/md0/files/cicids2018/ or change the path in prepocess.py.

Run in src/ :
```bash
    python3 main.py 1 1 1
```

Explanations of the numbers are given in the main.py file.
Please note that the preprocess.py file has a FEATURES_DROPPED list that can be commented in if the reduced model should be tested. Then also add the appropriate just use the combined model with frags in the main.py
