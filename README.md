# Overview

[[Blog]](https://openai.com/blog/dall-e/) [[Paper]](https://arxiv.org/abs/2102.12092) [[Model Card]](model_card.md) [[Usage]](notebooks/usage.ipynb)

This is the official PyTorch package for the discrete VAE used for DALL·E.

# Installation

Before running [the example notebook](notebooks/usage.ipynb), you will need to install the package using

	pip install git+https://github.com/openai/DALL-E.git

# Exporting VAE Weights

Run [export_weights.py](export_weights.py):

```
git clone https://github.com/shawwn/DALL-E
cd DALL-E
python3 export_weights.py
```

That will create files in your current directory like:


```
  encoder.blocks.input.w.shape.256.3.7.7
  encoder.blocks.input.b.shape.256
  encoder.blocks.group_1.block_1.res_path.conv_1.w.shape.64.256.3.3
  encoder.blocks.group_1.block_1.res_path.conv_1.b.shape.64
  encoder.blocks.group_1.block_1.res_path.conv_2.w.shape.64.64.3.3
  encoder.blocks.group_1.block_1.res_path.conv_2.b.shape.64
  ...  

  decoder.blocks.group_1.block_1.id_path.b.shape.2048
  decoder.blocks.group_1.block_1.id_path.w.shape.2048.128.1.1
  decoder.blocks.group_1.block_1.res_path.conv_1.b.shape.512
  decoder.blocks.group_1.block_1.res_path.conv_1.w.shape.512.128.1.1
  decoder.blocks.group_1.block_1.res_path.conv_2.b.shape.512
  decoder.blocks.group_1.block_1.res_path.conv_2.w.shape.512.512.3.3
  ...  
```

Each file is an array of floats. For example, the file `encoder.blocks.input.b.shape.256` contains exactly `256 * sizeof(float)` bytes of data.

You can read each file into whatever framework you want, reshaping the array to the shape specified in the filename. For example, `encoder.blocks.group_1.block_1.res_path.conv_1.w.shape.64.256.3.3` should be treated as a float32 tensor of shape `[64,256,3,3]`.

