Understanding Multi-Level Tagger
==============================

This repo is based on the paper [Zero-shot Sequence Labeling: Transferring Knowledge from Sentences to Tokens](https://arxiv.org/pdf/1805.02214.pdf) by Marek Rei and Anders Søgaard. The original code repositary can be found [here](https://github.com/marekrei/mltagger). We are extremely grateful to the authors for their work and an opportunity to go deep into understanding their paper.

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

This is done by `model.py`:
> count_interesting_labels = numpy.array([1.0 if batch[i][j][-1] != self.config["default_label"] else 0.0 for j in range(len(batch[i]))]).sum()

Notebooks
-------------------------
The `notebooks` folder contains jupyter notebooks packed with comments to help starters (like myself) get through the code. Feel free to put in issues or send in pull request if you spot any bugs.


Configuration
-------------------------

Edit the values in config.conf as needed:

* **path_train** - Path to the training data, in CoNLL tab-separated format. One word per line, first column is the word, last column is the label. Empty lines between sentences.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
* **default_label** - The most common (negative) label in the dataset. For example, the correct label in error detection or neutral label in sentiment detection.
* **model_selector** - What is measured on the dev set for model selection. For example, "dev_sent_f:high" means we're looking for the highest sentence-level F score on the development set.
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **emb_initial_zero** - Whether word embeddings should be initialized with zeros. Otherwise, they are initialized randomly. If 'preload_vectors' is set, the initialization will be overwritten either way for words that have pretrained embeddings.
* **train_embeddings** - Whether word embeddings are updated during training.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **hidden_layer_size** - Final hidden layer size, right before word-level predictions.
* **char_hidden_layer_size** - Char-level representation size, right before it gets combined with the word embeddings.
* **lowercase** - Whether words should be lowercased.
* **replace_digits** - Whether all digits should be replaced by zeros.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **singletons_prob** - The probability with which words that occur only once are replaced with OOV during training.
* **allowed_word_length** - Maximum allowed word length, clipping the rest. Can be necessary if the text contains unreasonably long tokens, eg URLs.
* **max_train_sent_length** - Discard sentences in the training set that are longer than this.
* **vocab_include_devtest** - Whether the loaded vocabulary includes words also from the dev and test set. Since the word embeddings for these words are not updated during training, this is equivalent to preloading embeddings at test time as needed. This seems common practice for many sequence labeling toolkits, so I've included it as well. 
* **vocab_only_embedded** - Whether to only include words in the vocabulary if they have pre-trained embeddings.
* **initializer** - Method for random initialization
* **opt_strategy** - Optimization methods, e.g. adam, adadelta, sgd.
* **learningrate** - Learning rate
* **clip** - Gradient clip limit
* **batch_equal_size** - Whether to construct batches from sentences of equal length.
* **max_batch_size** - Maximum batch size.
* **epochs** - Maximum number of epochs to run.
* **stop_if_no_improvement_for_epochs** - Stop if there has been no improvement for this many epochs.
* **learningrate_decay** - Learning rate decay when performance hasn't improved.
* **dropout_input** - Apply dropout to word representations.
* **dropout_word_lstm** - Apply dropout after the LSTMs.
* **tf_per_process_gpu_memory_fraction** - Set 'tf_per_process_gpu_memory_fraction' for TensorFlow.
* **tf_allow_growth** - Set 'allow_growth' for TensorFlow
* **lmcost_max_vocab_size** - Maximum vocabulary size for the language modeling objective.
* **lmcost_hidden_layer_size** - Hidden layer size for LMCost.
* **lmcost_lstm_gamma** - LMCost weight
* **lmcost_joint_lstm_gamma** - Joint LMCost weight
* **lmcost_char_gamma** - Char-level LMCost weight
* **lmcost_joint_char_gamma** - Joint char-level LMCost weight
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