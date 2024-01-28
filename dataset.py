import requests
import torch
import queue
import asyncio
import time

from transformers import AutoTokenizer

loop = asyncio.get_event_loop()

q = queue.Queue()

tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
tokenizer.add_special_tokens({
    'bos_token' : '<|s|>',
    'eos_token' : '<|e|>',
    'pad_token' : '<|p|>'
})

tokenizer.add_bos_token = True
tokenizer.add_pad_token = True
tokenizer.add_eos_token = True


from config import get_config
cfg = get_config()

def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0  #returns lower triangle with diagonal as True


def fetch_data():
    #print("fetch_data")
    url = cfg.get('feed_url')
    resp = requests.get(url)
    if resp.status_code == 200:
        result = resp.json()
        count = len(result)

        if count > 0:
            for i in range(count):
                q.put( result[i] )


    #loop.call_later(2, fetch_data)

fetch_data()


def fetch_batch():
    global q

    while cfg.get('micro_batch_size') > q.qsize():
        fetch_data()

    #if cfg.get('micro_batch_size') > q.qsize():
    #    return None

    inputs = tuple()
    labels = tuple()
    masks = tuple()

    for i in range(cfg.get('micro_batch_size')):
        text = q.get()
        q.task_done()

        token_ids = tokenizer.encode( text )
        if len(token_ids) > cfg.get('seq_len') - 1:
            token_ids = token_ids[: cfg.get('seq_len') - 1]

        padding_token_count = cfg.get('seq_len') - len(token_ids) - 1

        decoder_input = torch.cat(
            [
                torch.tensor([tokenizer.bos_token_id], dtype=torch.int64),
                torch.tensor(token_ids, dtype=torch.int64),
                torch.tensor([tokenizer.pad_token_id] * padding_token_count, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(token_ids, dtype=torch.int64),
                torch.tensor([tokenizer.eos_token_id], dtype=torch.int64),
                torch.tensor([tokenizer.pad_token_id] * padding_token_count, dtype=torch.int64)
            ],
            dim=0
        )

        inputs += (decoder_input, )
        labels += (label, )
        masks += ( (decoder_input != tokenizer.pad_token_id).int(), )




    return dict(
        input=torch.stack(inputs),
        label=torch.stack(labels),
        mask=torch.stack(masks)
    )
