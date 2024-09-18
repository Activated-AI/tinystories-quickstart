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

And then after an hour:

> Lily went to the park and saw a friendly dog. She wanted to play with the dog, but the dog was big and scary. Lily was scared and didn't want to go near the dog. She ran away and found her mommy on a bench. 

> "Mommy, the dog is scary," said Lily. "I don't want to play with him." 

> "Don't worry, Lily," said her mommy. "The dog is friendly. Come, let's go feed the dog." 

> Lily slowly approached the dog and saw that he wanted a treat. She gave the dog the treat and he wagged his tail happily. From that day on, Lily wasn't scared of dogs anymore and they became friends.

## Contributing

Feel free to fork this repository and submit pull requests for any improvements or features you'd like to add.

