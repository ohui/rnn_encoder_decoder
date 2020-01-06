## BiDirectional RNN Encoder Decoder 

These results are heavily inspired by the following two links: 

[How to Develop a Neural Machine Translation System from Scratch](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/)

[A Must-Read NLP Tutorial on Neural Machine Translation – The Technique Powering Google Translate](https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/)


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
