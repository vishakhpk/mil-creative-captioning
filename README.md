## Machine-in-the-Loop Rewriting for Creative Image Captioning

## Data

Annotated sources of data used in the paper:

| Data Source | URL |
| ----------- | --- |
| [Mohammed et al.](https://aclanthology.org/S16-2003/) | [Link](http://saifmohammad.com/WebPages/metaphor.html) |
| [Gordon et al.](https://aclanthology.org/W15-1407/) | [Link](https://www.isi.edu/people/mics/corpus_rich_metaphor_annotation) |
| [Bostan et al.](https://aclanthology.org/2020.lrec-1.194/) | [Link](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/goodnewseveryone/) |
| [Niculae et al.](https://aclanthology.org/D14-1215/) | [Link](http://vene.ro/figurative-comparisons/) |
| [Steen et al.](https://benjamins.com/catalog/celcr.14) | [Link](http://www.vismet.org/metcor/search/showPage.php?page=start) |

#### TODO: Individual data cleaning scripts

## Model Training

Follow the README in the model\_training directory to train a Fairseq BART model. [Reach out](mailto:vishakh@nyu.edu) for our trained model.
#### TODO: Upload model checkpoint to drive

## Interface

Code to run the UI we used for interactive experiments. This UI hosts a server and needs you to have a backend GPU to run model inference during interaction. The code saves each interaction with a unique ID which we use to match to our crowdworkers for experimental analysis. 

#### TODO: Data Processing Scripts to filter results 

