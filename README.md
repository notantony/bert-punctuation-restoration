# bert-punctuation-restoration

This project mainly based on the paper [1], trying to reproduce some of the results on TED Talks dataset.

## Usage

### Environment setup
Tested on Python 3.9.12.

Install required packages using Pip.\
`pip install -r reqiurements.txt`

Configure Python interpreter path with the following line.\
`export PYTHONPATH="$PYTHONPATH:./src"`

### Downloads
Download data using the script.\
`python ./src/data_utils/download_data.py`


If needed, download trained models and place it into `./models`\
Use one of the following links.
| name  | Google Drive | description |
|-------|--------------|------------|
|`base-256`|  [link](https://drive.google.com/file/d/1-0WolA-FZVVo22ZEyFGDZfEnKXexK562/view?usp=sharing) | BERT-base-uncased + 2-layer dense classifier with 256 hidden dimensions on top| 


### References
1. Nagy, Attila, Bence Bial, and Judit Ács. "Automatic punctuation restoration with BERT models." arXiv preprint arXiv:2101.07343 (2021).