from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizer.trainers import WordLevelTrainer
from tokenizer.pre_tokenizers import WhiteSpace
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

        
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file']).format(lang)
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = WhiteSpace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.saved(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config['lang_scr']}-{config['lang_tgt']}', split = 'train')
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_scr'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    training_split = 0.9 * (len(ds_raw))
    validation_split = len(ds_raw) - training_split
    train_data, validation_data = random_split(ds_raw, [training_split, validation_split])
