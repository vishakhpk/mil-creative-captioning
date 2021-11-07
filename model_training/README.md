## Code to train the BART Model

This code is minorly changed from this the Fairseq [example](https://github.com/pytorch/fairseq/blob/main/examples/bart/README.summarization.md).

1. The code expects files in the (train.source, train.target, val.source, val.target) format commonly found with Fairseq. The first steps are preprocessing. Set the TASK variable in the scripts to the directory where the data files are located. 
```
sh bpe_encoder.sh
sh binarize_dataset.sh
``` 
2. Fine-tune the BART model on the data. The preprocessing step creates a new directory with the binarized data files which is consumed by this train script.
```
sh train.sh
``` 
