import torch


class BertDataset:
    def __init__(self, text, tokenizer, max_len, target=None):
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        texts = self.text[idx].split('\n')
        output = self.tokenizer.encode_plus(
            " ".join(str(texts[0]).split()),
            " ".join(str(texts[1]).split()),
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = output['input_ids']
        masks = output['attention_mask']
        token_type_ids = output['token_type_ids']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.target[idx], dtype=torch.float)
        }