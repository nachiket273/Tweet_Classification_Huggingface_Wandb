# Tweet_Classification_Huggingface_Wandb
Simple Repository for kaggle competition regarding [tweet classification](https://www.kaggle.com/c/nlp-getting-started).
This repository uses [huggingface](https://huggingface.co/welcome) tokenizer and transformer model which case specified 
as input argument, and tracks accuracy, losses and gradients using [wandb](https://www.wandb.com/).

# Configuration
config.py contains all the default configuration parameters.
They are used by code to create and load model, load train and test data, decide dropout ratio etc.
Look through [config.py](./config.py) for more information.

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
