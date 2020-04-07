# Tweet_Classification_Huggingface_Wandb
Simple Repository for kaggle competition regarding [tweet classification](https://www.kaggle.com/c/nlp-getting-started).
This repository uses [huggingface](https://huggingface.co/welcome) tokenizer and transformer model which case specified 
as input argument, and tracks accuracy, losses and gradients using [wandb](https://www.wandb.com/).

# Configuration
config.txt contains all the default configuration parameters.
They are used by code to create and load model, load train and test data, get batch size,
decide dropout ratio etc.
Look through [config.txt](./config.txt) for more parameters.

*Defaults*
```
[DEFAULT]
start_lr = 3e-5
train_bs = 8
valid_bs = 8
epochs = 5
max_len = 160
dropout_ratio = 0.1
linear_in = 768
num_classes = 2
warmup_epochs = 0
test_size = 0.2
train_file ='./train.csv'
test_file = './test.csv'
model_name = 'bert-base-uncased'
seed = 42
use_sched = True
```

Use get_config and set_config from [config.py](./config.py) to read and update config.txt.
set_config accepts dictonary to set new values for parameters.

*Note:-* get_config and set_config use configparser to get and set config, and config.txt
adheres to file structure expected by configparser.

# Usage

*Arguments*:
```Python
    --freeze             : If true, all layers except top linear layers will be freezed.
                           Default: True 
    --save_plot          : Save loss and accuracy plots.
                           Default: False
    --track              : Track the stats using wandb.
                           Default: False
    --wandb_project_name : Name of wandb project.
```

*Example*:
```python
# Default
python main.py

# With unfreezed layers and saving plots.
python main.py --freeze False --track True

# Track using wandb
python main.py --track True --wandb_project_name <name_of_project>
```
*NOTE:-* When '--track' is True, program expects wandb API key to be set
through enviornment variable 'WANDB_API_KEY'.

# Dependencies:

* [transformers](https://github.com/huggingface/transformers)
* [wandb](https://www.wandb.com/)

These libraries can be installed through pip
```
pip install transformers
pip install wandb
```

# Results:

Will update links soon.
