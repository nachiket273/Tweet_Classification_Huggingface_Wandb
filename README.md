# Tweet_Classification_Huggingface_Wandb
Simple Repository for kaggle competition regarding [tweet classification](https://www.kaggle.com/c/nlp-getting-started).
This repository uses huggingface tokenizer and transformer model which case specified as input argument, and tracks accuracy, losses and gradients using [wandb](https://www.wandb.com/).

# Usage

Arguments:
```
    --start_lr:  Initial Learning rate
                 default=3e-5, type=float
    --train_bs:  Batch Size for training
                 default=8, type=int
    --valid_bs:  Batch Size for validation
                 default=4, type=int
    --epochs:    Number of  training 
                 epochs default=1, type=int
   --model_name: [Pretrained Bert Model name to be loaded 
                 from huggingface transformer library](https://huggingface.co/transformers/pretrained_models.html)
                 type=str
   --train_file: Path to train csv file.
                 type=str
   --test_file:  Path to test csv file.
                 type=str
   --max_len:    Total token sequence length to be returned by tokenizer.
                 default=512, type=int
   --dropout_ratio: Dropout ratio
                    default=0.3, type=float
   --num_classes: Number of classification labels/classes
                  default=2, type=int
   --warmup_epochs: Number of epochs to be used to provide 
                    warmup steps to transfomer.
                    default=0, type=int
   --plot_stats: Plot accuracy, loss plots.
                 default=False, type=bool
   --preprocess: Preprocess the text before training/predictions
                 default=False, type=bool
   --use_keyword: Use Keyword field with text field
                  default=False, type=bool
   --wandb_project_name: wandb project name 
                         type=str
   --wandb_key_file: Path to file containing wandb api key
                     Make sure file is secure and read-only.
                     type=str
   --freeze:  If true, all other layers other than top linear layer
              will be freezed. 
              default=True, type=bool
```

Example:
```python
python main.py --start_lr 3e-5 --train_bs 8 --valid_bs 4 --epochs 5 --model_name 'bert-base-uncased' \
--train_file './train.csv' --test_file './test.csv' --max_len 90 --dropout_ratio 0.5 --num_classes 2 \
--warmup_epochs 1 --plot_stats True --preprocess True --use_keyword False \
--wandb_project_name 'tweet_classification_huggingface_bert' --wandb_key_file './api_key'
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
