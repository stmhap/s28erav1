from dataset import fetch_batch, tokenizer, cfg
import torch
import torch.optim as optim

from phi2.modeling_phi import PhiForCausalLM
from phi2.configuration_phi import PhiConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_config = PhiConfig(
    vocab_size=len(tokenizer),
    causal=True
)
model = PhiForCausalLM(model_config)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()

total_micro_batches = 0
epoch = 0

step_count = -1
epoch_loss = []

print('---->>>>> Training logs <<<<<-----')
while True:
    micro_batch = fetch_batch()

    if micro_batch is None:
        continue

    output = model(
        input_ids=micro_batch['input'].to(device),
        attention_mask=micro_batch['mask'].to(device),
        labels=micro_batch['label'].to(device)
    )

    loss = output['loss']
    loss.backward()

    if step_count == -1:
        print('Epoch:', '%04d' % (epoch), 'Step count: ', step_count, 'loss =', '{:.6f}'.format(loss.item()))
        step_count += 1

    epoch_loss.append(loss.item())

    total_micro_batches += micro_batch['input'].size(0)
    #print('Total micro batches: ', total_micro_batches)
    if total_micro_batches % cfg.get('batch_size') == 0:
        #print('Step backward: ', True)
        optimizer.step()
        optimizer.zero_grad()
        step_count += 1

        #a = torch.tensor(epoch_loss, dtype=torch.float32)
        print('\t\t', 'Step count: ', step_count, 'loss =', '{:.6f}'.format(loss.item()))

        if step_count % cfg.get('epoch_steps') == 0:
            b = torch.tensor(epoch_loss, dtype=torch.float32)
            print('Epoch:', '%04d' % (epoch + 1), 'Step count: ', step_count, 'loss =', '{:.6f}'.format(b.mean()))
            epoch += 1
            epoch_loss = []



