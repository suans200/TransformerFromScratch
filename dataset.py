import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, seq_len:int, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sos_tokenized = torch.tensor([self.src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_tokenized = torch.tensor([self.src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_tokenized = torch.tensor([self.src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    
    def __len__(self, ds):
        return len(ds)
    def __get_item__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.tgt_tokenizer.encode(tgt_text).ids
        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1
        if enc_num_padding<0 or dec_num_padding<0:
            raise ValueError("Sentence is too long.")
        
        encoder_input = torch.cat(
            [
                self.sos_tokenized,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_tokenized,
                torch.tensor([self.pad_tokenized]*enc_num_padding, dtype=torch.int64)
            ],
            dim=0
        )
        decoder_input = torch.cat(
            [
                self.sos_tokenized,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_tokenized]*dec_num_padding,dtype=torch.int64)
            ],
            dim=0
        )

        labels = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_tokenized,
                torch.tensor([self.pad_tokenized]*dec_num_padding, dtype=torch.int64)
            ]
        )
        
        assert labels.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert encoder_input.size(0) == self.seq_len
        return {
            "encoder_input": encoder_input, 
            "decoder_input": decoder_input,  
            "encoder_mask": (encoder_input != self.pad_tokenized).unsqueeze(0).unsqueeze(0).int(), 
            "decoder_mask": (decoder_input != self.pad_tokenized).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), 
            "label": labels,  
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0