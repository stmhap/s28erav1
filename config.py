from pathlib import Path


def get_config():
    return dict(
        micro_batch_size=4,
        batch_size=8,   #should be multiple of micro_batch
        epoch_steps=10,
        seq_len=512,
        preload=False,
        feed_url='https://e60f-35-233-217-40.ngrok-free.app/generate'
    )