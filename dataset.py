import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilinguialDataset(Dataset):
    def __init__(self, ds, src_tokenizer, traget_tokenizer, src_language, target_language, sequence_len):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = traget_tokenizer
        self.src_language = src_language
        self.target_language = target_language
        self.sequence_len = sequence_len
        self.ds = ds
        self.sos_token = torch.tensor([self.target_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.target_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.target_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_language]
        target_text = src_target_pair["translation"][self.target_language]
        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokenz = self.target_tokenizer.encode(target_text).ids
        enc_num_padding_tokens = self.sequence_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.sequence_len - len(dec_input_tokenz) - 1
        if enc_num_padding_tokens<2 or dec_num_padding_tokens<2 :
            raise ValueError("Sentence is too long.")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim = 0
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokenz, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim = 0
        )

        label = torch.cat([
            torch.tensor(dec_input_tokenz, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.sequence_len
        assert decoder_input.size(0) == self.sequence_len
        assert label.size(0) == self.sequence_len

        return{
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "label" : label,
            "src_text" : src_text,
            "tgt_text" : target_text,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
        }
    
    def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0