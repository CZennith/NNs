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

def train_fn (
        model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)                    # src = [src length, batch size]
        trg = batch["en_ids"].to(device)                    # trg = [trg length, batch size]

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)     # output = [trg length, batch size, trg vocab size]
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)            # output = [(trg length - 1) * batch size, trg vocab size]

        trg = trg[1:].view(-1)                              # trg = [(trg length - 1) * batch size]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            src = batch["de_ids"].to(device)                # src = [src length, batch size]
            trg = batch["en_ids"].to(device)                # trg = [trg length, batch size]

            output = model(src, trg, 0)                     #no teacher forcing
                                                            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)        # output = [(trg length - 1) * batch size, trg vocab size]

            trg = trg[1:].view(-1)                          # trg = [(trg length - 1) * batch size]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def translate_sentence(
        sentence, model, en_nlp, de_nlp, en_vocab, de_vocab, lower, sos_token, eos_token, device, max_output_length=25
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)               #get their numbers
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            input_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(input_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)

            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens


def get_tokenizer_fn(nlp, lower):
    def tokenizer_fn(s):
        tokens = [token.text for token in nlp.tokenizer(s)]
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens
    return tokenizer_fn
