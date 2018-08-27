Understanding Multi-Level Tagger
==============================

This repo is based on the paper [Zero-shot Sequence Labeling: Transferring Knowledge from Sentences to Tokens](https://arxiv.org/pdf/1805.02214.pdf) by Marek Rei and Anders Søgaard. The original code repositary can be found [here](https://github.com/marekrei/mltagger). We ([my co-worker](https://github.com/fieryheart34) and I) are extremely grateful to the authors for their work and an opportunity to go deep into understanding their paper.

Referencing their code, we have attempted to understand the paper by breaking down the code into bite-sized information peppered with notes and comments. 

Getting Started
-------------------------
These are the few things you can do in this repo.

1. Read notebooks with code comments in `./notebooks`
2. Run experiment / model training with `python experiment.py config_file.conf`
    - This will start the training process using the TSV file `./conf/config_file.config:path_train` defined in the configuration file in the `./conf` folder. An output model will be stored in path defined in `./conf/config_file.config:save`.
3. Perform inference based on trained model with `python print_output.py ./saved_model/model.model input_file.tsv`
    - This will print the original file with two additional columns: the token-level score and the sentence-level score. The latter will be the same for all tokens in a sentence.
4. Try out the model over an interface at https://derekchia.github.io/mltagger

Directory Structure
-------------------------

    |- notebooks
        |- experiment.ipynb
        |- model.ipynb
        |- evaluator.ipynb
        |- print_output.ipynb
    |- data
        |- glove.6B.300d.txt
    |- conf
        |- config.conf
    |- saved_model
        |- fce_model.model
    |- experiment.py
    |- model.py
    |- evaluator.py
    |- print_output.py
    |- README.md

Input Data format
-------------------------

The training and test data is expected in standard CoNLL-type tab-separated format. One word per line, separate column for token and label, empty line between sentences.

For error detection, this would be something like:

    I       c
    saws    i
    the     c
    show    c
    
See the original dataset for error detection [here](https://ilexir.co.uk/datasets/index.html).

The binary word-level and sentence-level labels are constructed from this format automatically, based on the *default_label* value. Please **remember** to change the *default_label* before you run `experiment.py`.

Any word with *default_label* gets label 0, any word with other labels gets assigned 1.
Any sentence that contains only *default_label* labels is assigned a sentence-level label 0, any sentence containing different labels gets assigned 1. 

Referencing the use case of error detection, the correct word will be represented by label 0 and incorrect word will be assigned 1. If the entire sentence only contain default label (0 / correct), the sentence-level label will be 0 - indicating no error found in the sentence. Otherwise, if sentence contains word label(s) different from label 0, the sentence-level label will be 1. 

This label translation is done by the following line in `model.py`:
> count_interesting_labels = numpy.array([1.0 if batch[i][j][-1] != self.config["default_label"] else 0.0 for j in range(len(batch[i]))]).sum()

Notebooks
-------------------------
The `notebooks` folder contains jupyter notebooks packed with comments to help starters (like myself) get through the code. Feel free to put in issues or send in pull request if you spot any bugs.


Configuration
-------------------------

Edit the values in config.conf as needed:

* **path_train** - Path to the training data, in CoNLL tab-separated format. One word per line, first column is the word, last column is the label. Empty lines between sentences.
    - e.g. PATH_TO/fce-public.train.original.tsv
* **path_dev** - Path to the development data, used for choosing the best epoch.
    - e.g. PATH_TO/fce-public.dev.original.tsv
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
    - PATH_TO/fce-public.test.original.tsv
* **default_label** - The most common (negative) label in the dataset. For example, the correct label in error detection or neutral label in sentiment detection.
    - Defaults to *O*. **CHANGE THIS**
* **model_selector** - What is measured on the dev set for model selection. For example, "dev_sent_f:high" means we're looking for the highest sentence-level F score on the development set.
    - Defaults to *dev_sent_f:high*
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
    - e.g. PATH_TO/EMBEDDING.txt
* **word_embedding_size** - Size of the word embeddings used in the model.
    - Defaults to *300*
* **emb_initial_zero** - Whether word embeddings should be initialized with zeros. Otherwise, they are initialized randomly. If 'preload_vectors' is set, the initialization will be overwritten either way for words that have pretrained embeddings.
    - Defaults to *False*
* **train_embeddings** - Whether word embeddings are updated during training.
    - Defaults to *True*
* **char_embedding_size** - Size of the character embeddings.
    - Defaults to *100*
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
    - Defaults to *300*
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
    - Defaults to *100*
* **hidden_layer_size** - Final hidden layer size, right before word-level predictions.
    - Defaults to *50*
* **char_hidden_layer_size** - Char-level representation size, right before it gets combined with the word embeddings.
    - Defaults to *50*
* **lowercase** - Whether words should be lowercased.
    - Defaults to *True*
* **replace_digits** - Whether all digits should be replaced by zeros.
    - Defaults to *True*
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
    - Defaults to *-1*
* **singletons_prob** - The probability with which words that occur only once are replaced with OOV during training.
    - Defaults to *0.1*
* **allowed_word_length** - Maximum allowed word length, clipping the rest. Can be necessary if the text contains unreasonably long tokens, eg URLs.
    - Defaults to *-1*
* **max_train_sent_length** - Discard sentences in the training set that are longer than this.
    - Defaults to *-1*
* **vocab_include_devtest** - Whether the loaded vocabulary includes words also from the dev and test set. Since the word embeddings for these words are not updated during training, this is equivalent to preloading embeddings at test time as needed. This seems common practice for many sequence labeling toolkits, so I've included it as well. 
    - Defaults to *False*
* **vocab_only_embedded** - Whether to only include words in the vocabulary if they have pre-trained embeddings.
    - Defaults to *False*
* **initializer** - Method for random initialization
    - Defaults to *glorot*
* **opt_strategy** - Optimization methods, e.g. adam, adadelta, sgd.
    - Defaults to *adadelta*
* **learningrate** - Learning rate
    - Defaults to *1.0*
* **clip** - Gradient clip limit
    - Defaults to *0.0*
* **batch_equal_size** - Whether to construct batches from sentences of equal length.
    - Defaults to *False*
* **max_batch_size** - Maximum batch size.
    - Defaults to *32*
* **epochs** - Maximum number of epochs to run.
    - Defaults to *200*
* **stop_if_no_improvement_for_epochs** - Stop if there has been no improvement for this many epochs.
    - Defaults to *7*
* **learningrate_decay** - Learning rate decay when performance hasn't improved.
    - Defaults to *0.9*
* **dropout_input** - Apply dropout to word representations.
    - Defaults to *0.5*
* **dropout_word_lstm** - Apply dropout after the LSTMs.
    - Defaults to *0.5*
* **tf_per_process_gpu_memory_fraction** - Set 'tf_per_process_gpu_memory_fraction' for TensorFlow.
    - Defaults to *1.0*
* **tf_allow_growth** - Set 'allow_growth' for TensorFlow
    - Defaults to *True*
* **lmcost_max_vocab_size** - Maximum vocabulary size for the language modeling objective.
    - Defaults to *7500*
* **lmcost_hidden_layer_size** - Hidden layer size for LMCost.
    - Defaults to *50*
* **lmcost_lstm_gamma** - LMCost weight
    - Defaults to *0.0*
* **lmcost_joint_lstm_gamma** - Joint LMCost weight
    - Defaults to *0.0*
* **lmcost_char_gamma** - Char-level LMCost weight
    - Defaults to *0.0*
* **lmcost_joint_char_gamma** - Joint char-level LMCost weight
    - Defaults to *0.0*
* **char_integration_method** - Method for combining character-based representations with word embeddings.
    - Defaults to *concat*. Option of *concat* or *none*
* **save** - Path for saving the model.
    - Defaults to *None*. Please add in path to save model.
* **garbage_collection** - Whether to force garbage collection.
    - Defaults to *False*. Not in use.
* **lstm_use_peepholes** - Whether LSTMs use the peephole architecture.
    - Defaults to *False*
* **whidden_layer_size** - Hidden layer size after the word-level LSTMs.
    - Defaults to *200*
* **attention_evidence_size** - Layer size for predicting attention weights.
    - Defaults to *100*
* **attention_activation** - Type of activation to apply for attention weights.
    - Defaults to *soft*. Option of *sharp*, *soft*, *linear*
* **attention_objective_weight** - The weight for pushing the attention weights to a binary classification range.
    - Defaults to *0.01*
* **sentence_objective_weight** - Sentence-level objective weight.
    - Defaults to *1.0*
* **sentence_objective_persistent** - Whether the sentence-level objective should always be given to the network.
    - Defaults to *True*
* **word_objective_weight** - Word-level classification objective weight.
    - Defaults to *0.0*
* **sentence_composition** - The method for sentence composition.
    - Defaults to *attention*. Option of *last* or *attention*
* **random_seed** - Random seed.
    - Defaults to *100*