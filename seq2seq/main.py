import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate

from model import Encoder, Decoder, Seq2Seq
from training import train_fn, evaluate_fn, translate_sentence, get_tokenizer_fn

seed = 1234

#setting seeds to be random 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


#retrieving datasets
dataset = datasets.load_dataset("bentrevett/multi30k")

train_data, valid_data, test_data = (
    dataset["train"],
    dataset["validation"],
    dataset["test"],
)


#tokenizing (turning a string into a list of words and punctuation, considered tokens)

#these are tokenizing models
en_nlp = spacy.load("en_core_web_sm")
de_nlp = spacy.load("de_core_news_sm")


# #example of what it does
# string = "What a lovely day it is today!"
# for token in en_nlp.tokenizer(string):
#     print(token)                            #seperates the string into individual tokens(words and punctuation)

def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]

    #make them lower case if needed
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]

    #add sos and eos tokens to indicate the start and end
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]

    return {"en_tokens": en_tokens, "de_tokens": de_tokens}

max_length = 1000
lower = True
sos_token = "<sos>"
eos_token = "<eos"

#store argument values in a dictionary to use in the .map() below
fn_kwargs = {
    "en_nlp": en_nlp,
    "de_nlp": de_nlp,
    "max_length": max_length,
    "lower": lower,
    "sos_token": sos_token,
    "eos_token": eos_token,
}

train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)


#creating vocabularies (giving different words an index / number, if they appear more than the min_frequency) with torchtext
min_freq = 2
unk_token = "<unk>"
pad_token = "<pad>"

#to registere within the model evne if they have a freq below min_freq as they are needed
special_token = [
    unk_token,
    pad_token,
    eos_token,
    sos_token,
]

en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq = min_freq,
    specials = special_token,
)

de_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["de_tokens"],
    min_freq = min_freq,
    specials = special_token,
)

#printing what's in the voacb. get_itos() converts Int TO String allowing us to see the tokenized form of the numerical elements
# print(en_vocab.get_itos()[:10])
# print(en_vocab.get_itos()[9])
# print(len(en_vocab), len(de_vocab))

assert(en_vocab[unk_token] == de_vocab[unk_token])
assert(en_vocab[pad_token] == de_vocab[pad_token])

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

#default unkown / unrefistered tokens to be 0 (unk index)
en_vocab.set_default_index(unk_index)
de_vocab.set_default_index(unk_index)

def numericalize_example(example, en_vocab, de_vocab):

    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}

fn_kwargs = {"en_vocab":en_vocab, "de_vocab":de_vocab}

train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)



data_type = "torch"
format_columns = ["en_ids", "de_ids"]

#format data's integers into pytorch tensors for model to read later on
train_data = train_data.with_format(
    type = data_type,
    columns = format_columns,
    output_all_columns = True,
)

valid_data = valid_data.with_format(
    type = data_type,
    columns = format_columns,
    output_all_columns = True,
)

test_data = test_data.with_format(
    type = data_type,
    columns = format_columns,
    output_all_columns = True,
)

#by returning a function within a function, collate fn can use the padindex without crating a global vlauye or clas
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        
        #pad every tensor valye to be equal to the longerst one
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch
    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 128

#only the training data is shuffled tomake the training data more random and increase accuracy dyuring traiing
train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


#training the models initialised in model.py
input_dim = len(de_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")            #prints number of trainable parameters

#used to update parameters in our training loop
optimizer = optim.Adam(model.parameters())

#getting the Loss via cross entropy loss
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)                             #ignores the loss when the token is = to padding token


n_epochs = 10
clip = 1.0          #where the gradient is clipped to prevent exploding gradient
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):

    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )

    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")


    print(f"\t Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\t Valid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")


# model.load_state_dict(torch.load("tut1-model.pt"))
# test_loss = evaluate_fn(model, test_data_loader,criterion, device)
# print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

translations = [translate_sentence(
    example["de"], 
    model, 
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device
) for example in tqdm.tqdm(test_data)
]

bleu = evaluate.load("bleu")

predictions = [" ".join(translation[1:-1]) for translation in translations]             #join translated words into string
references = [[example["en"]] for example in test_data]                                 #answer

tokenizer_fn = get_tokenizer_fn(en_nlp, lower)
# print(tokenizer_fn(predictions[0]), tokenizer_fn(references[0][0]))

#calcualte results with bleu
results = bleu.compute(
    predictions=predictions, references=references, tokenizer=tokenizer_fn
)

print(results)

