import transformers
import torch.nn as nn

class BertHf(nn.Module):
    '''
    Simple Model with uncased bert pretrained model.
    '''
    def __init__(self, model_name, dp=0.3, num_classes=1, linear_in=768):
        super(BertHf, self).__init__()
        self.model = transformers.BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dp)
        self.out = nn.Linear(linear_in, num_classes)

    def forward(self, ids, masks, token_type_ids):
        out1, out2 = self.model(ids, attention_mask=masks, token_type_ids=token_type_ids)
        out2 = self.dropout(out2)
        out2 = self.out(out2)
        return out2
        