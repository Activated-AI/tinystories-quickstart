# tinystories-quickstart

A quick, easy, and hackable way to train and evaluate performant TinyStories models.

## Quick Start

```bash
pip install transformers datasets
python gpt.py
```

## Overview

This project started from [Andrej Karpathy's NanoGPT](https://github.com/karpathy/build-nanogpt), slimmed down to a single GPU and optimized for quick training runs on [TinyStories](https://arxiv.org/abs/2305.07759).

## Features

- Downloads a dataset and tokenizer from Hugging Face on the first run
- Caches your data for quick future runs
- Optimized for single GPU training

## Example Output

After 10 minutes of training on an H100 GPU, the model completes "Lily went to the park and saw a friendly dog." with the following:


> Lily went to the park and saw a friendly dog. The dog was wagging its tail and sniffing around a nice puddle. Lily realized that the water was clean and shiny. She started to dance around and make up a little song. The dog barked and joined in on the dance. Lily had so much fun playing with the dog and forgot about the time. 
> 
> As the sun began to set, Lily was sad to leave the park, but she knew she could come back again and dance with the dog again. She smiled and held on tight to the umbrella as she walked home.


## Contributing

Feel free to fork this repository and submit pull requests for any improvements or features you'd like to add.

