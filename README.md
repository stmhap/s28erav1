# Capstone Project: Part 1 (S28 ERA V1)
## Train LLM from scratch. ##

* You need to select a model that is less than 3B parameters (can be Microsoft's Phi 2 as well, but with random weights, hence training logs are MUST for capstone)
* Data:
  * It would be close to impossible to collect ALL the datasets required to train your model. Hence:
  * You are going to use Microsoft's Phi-2 or any other model and generate data. Recommend you generate this data in parallel, don't generate and store everything as that would be a very very large dataset
  * You are going to collect "some" clean data (100MB when zipped). This data CAN be generated from Phi-2 and stored.
* Training
  * You are going to use the same tokenizer and other data structures to keep your life simple
  * You are going to use AWS (or an equvalent system) where you are going to train YOUR model. 
  * You are going to train YOUR model (let's say that starts at 10). Train it somehow to reach the "initial loss - 1" value. Compare it with the final Microsoft's Phi 2's value and see how much more you have to train!!!

### Training Logs ###
```
Epoch: 0000 Step count:  -1 loss = 11.101167
		 Step count:  1 loss = 11.078608
		 Step count:  2 loss = 6.995222
		 Step count:  3 loss = 8.732363
		 Step count:  4 loss = 10.983467
		 Step count:  5 loss = 7.682126
		 Step count:  6 loss = 6.839418
		 Step count:  7 loss = 8.480166
		 Step count:  8 loss = 7.619623
		 Step count:  9 loss = 7.381527
		 Step count:  10 loss = 7.133785
Epoch: 0001 Step count:  10 loss = 8.751073
		 Step count:  11 loss = 7.391686
		 Step count:  12 loss = 9.393476
		 Step count:  13 loss = 9.144436
		 Step count:  14 loss = 8.941932
		 Step count:  15 loss = 8.992459
		 Step count:  16 loss = 3.553864
		 Step count:  17 loss = 8.735229
		 Step count:  18 loss = 8.871849
		 Step count:  19 loss = 7.911204
		 Step count:  20 loss = 8.087790
Epoch: 0002 Step count:  20 loss = 7.917281
		 Step count:  21 loss = 8.564671
		 Step count:  22 loss = 5.447293
		 Step count:  23 loss = 6.475473
		 Step count:  24 loss = 6.703707
		 Step count:  25 loss = 8.246754
		 Step count:  26 loss = 7.081114
		 Step count:  27 loss = 4.978745
		 Step count:  28 loss = 7.537660
		 Step count:  29 loss = 8.182168
		 Step count:  30 loss = 8.499785
Epoch: 0003 Step count:  30 loss = 6.702834
```

### Clean Dataset
Use clean dataset from here. Need to rename sample.mp4 to sample.zip
to use it.
[https://www.kaggle.com/datasets/medihemaap/transformer-clean-dataset-sample/](https://www.kaggle.com/datasets/medihemaap/transformer-clean-dataset-sample/data)

### data-feed-server.ipynb
This Jupyter notebook is responsible for generating data-feed for our model training.
The following piece of code either generates text from a pre-trained transformer (Phi-2) model or reads randomly from sample.zip file
every second and keeps on adding it to a Queue asynchronously.

```
loop = asyncio.get_event_loop()
import queue
q = queue.Queue()

def callback(flag=False):
    print("Adding Text To Queue", flag)
    if flag:
        idx = random.randint(1, tokenizer_length)
        token = tokenizer.decode([idx])
        q.put( generate_text(token) )
        #q.put( read_text() )
    else:
        q.put( read_text() )
    
    
    if q.qsize() > 100:
        flag = True
    else:
        flag = False
    
    loop.call_later(0.01, callback, flag)
    #callback()
    
            
callback()
```

And the following code snippet starts an ngrok-web-server, and exposes server \<Public URL\>

```
port = 8000
ngrok_tunnel = ngrok.connect(port, pyngrok_config=conf.PyngrokConfig(auth_token=Ngrok_token))

# where we can visit our fastAPI app
print('Public URL:', ngrok_tunnel.public_url)


nest_asyncio.apply()

# finally run the app
uvicorn.run(app, port=port)
```

**Note:** <ins>_The \<Public URL\> changes every time the code is run_</ins>

```
INFO:     Started server process [127]
INFO:     Waiting for application startup.
Public URL: https://e60f-35-233-217-40.ngrok-free.app
```

when \<Public URL\>/generate API is called it returns an array of texts.
These texts are either from the stored text-files (sample.zip) or generated from a pre-trained Phi-2 model.

## training.ipynb
This training notebook is run on colab to train the model. please note that \<Public URL\>/generate needs to be added as **feed_url** in
_config.py_ file 

