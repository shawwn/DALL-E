"""Setup instructions:

git clone https://github.com/openai/DALL-E
cd DALL-E
wget https://cdn.openai.com/dall-e/encoder.pkl
wget https://cdn.openai.com/dall-e/decoder.pkl
python3

"""

import torch # if this errors, make sure torch is installed: python3 -m pip install torch


encoder = torch.load('encoder.pkl')
decoder = torch.load('decoder.pkl')

for name, tensor in encoder.state_dict().items():
  weights = tensor.numpy();
  filename = 'encoder.' + name + '.shape.' + '.'.join([str(x) for x in weights.shape])
  print(filename)
  weights.tofile(filename)

for name, tensor in decoder.state_dict().items():
  weights = tensor.numpy();
  filename = 'decoder.' + name + '.shape.' + '.'.join([str(x) for x in weights.shape])
  print(filename)
  weights.tofile(filename)
