## BiDirectional RNN Encoder Decoder 

These results are heavily inspired by the following two links: 

[How to Develop a Neural Machine Translation System from Scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)

[A Must-Read NLP Tutorial on Neural Machine Translation – The Technique Powering Google Translate](https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/)

## Notes: 

One of the hardest thing for me was really learning what `return_sequences` and `return_states`. 

### Return_sequences (default: False):

If you are stacking LSTMs together, you must enable `return_sequences` in each LSTM unit **if it is before another LSTM unit** so your last LSTM unit may not have it. Why? Personally, the way I look at it, I see that a LSTM unit requires a "sequence" of input (`( batch_size, time_steps, seq_len)`). It processes a sequence and may either return a 2d `(batch_size, units)` output when `Return_sequences= True` . or 3d `(batch_size, time_steps, units)` when `Return_sequences= True`. If you're sending something a LSTM unit, you need the 3d input, thus any LSTM unit before another LSTM unit must have `Return_sequences= True`. 

### Return_state (default: False):

LSTMS don't normally need to output its hidden state, but we may want that latent representation as output of a encoder and then input that into a decoder in the encoder decoder neural machine translation (as done in the notebook here). It's interesting because it's a 3d output that repeats the hidden state twice. 

> &nbsp;1. The LSTM hidden state output for the last time step. <br> &nbsp;2. The LSTM hidden state output for the last time step (again).<br>
     &nbsp;3. The LSTM cell state for the last time step.
     
[(source)](https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/)

But it makes sense when you set `return_sequences=True, return_state=True` and now the 1st item is the returned sequence (and 2nd itme is last hidden state, which is the last item in the returned sequence). 


## Summary 
This is a German to English neural machine translator trained on data from [ManyThings.org](http://www.manythings.org/anki/) using a simple encoder-decoder model. For each sentence, a single vector of integer encoded values for each word is used instead of one-hot encoding. The loss function is called `sparse_categorial_crossentropy` and it saves memory and computation compared to `catagorical_crossentropy`. I have not read any downsides of doing things this way. I initially used a unidirectional encoder, and then a bidirectional one after. Results show that bidirectional RNN does better than unidirecitonal RNN, but only by a little bit, which is not surprising.

## Results: 

Here are my weight for the BLEU score [(Source)](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/): 
```
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25,
```

### UniDirectional

| | Training data | Testing data |
|-- | ------------- | ------------- |
|BLEU-1 | 0.951738 | 0.566483  |
|BLEU-2 | 0.935278  | 0.453349  |
|BLEU-3 | 0.857045  | 0.371153  |
|BLEU-4 | 0.533102  | 0.178195* |

*area where uni did bettter

### BiDirectional

| | Training data | Testing data |
|-- | ------------- | ------------- |
|BLEU-1 | 0.953135 | 0.577633  |
|BLEU-2 | 0.937863  | 0.460237  |
|BLEU-3 | 0.860180  | 0.374129  |
|BLEU-4 | 0.536115  | 0.177248  |

## Improvements to be made

- Try out Teacher Forcing technique 
- Include attention 
- Compare against one-hot results 
