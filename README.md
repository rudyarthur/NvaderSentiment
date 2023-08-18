# NvaderSentiment
Vader sentiment with bootstraps
![example](https://github.com/rudyarthur/NvaderSentiment/blob/main/vaderSentiment/output.png)
This code is adapted from the popular [vaderSentiment](https://github.com/cjhutto/vaderSentiment/tree/master) library. You use it in the same way but it has a few interesting new features
```
from NvaderSentiment import NSentimentIntensityAnalyzer
Nvader = NSentimentIntensityAnalyzer(lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt", Nboots=1, normtype='standard')
s = "Nvader is not very smart." 
print(Nvader.polarity_scores(s, return_tagged=False))
```
Default behaviour is identical to VADER.

**1. flags**
```
apply_boost
apply_allcap 
apply_negation 
apply_punctemph 
apply_but
apply_specialcases
```

These can be used to turn off the various rules VADER uses when assigning sentiment. Turn them all off with `Nvader.no_modify()`, turn off everything but negation with `Nvader.only_negate()` otherwise pick and choose which ones to use.

**2. normalization**

 - Set `normtype='standard'` to use the usual rule for turning summed word scores into a polarity value `p = s/sqrt(s*s+alpha)` where `alpha=15` usually
 - Set `normtype='raw'` to return the raw summed word scores
 - Set `normtype='linear'` to use the rule `p = maxv*s/alpha` here `alpha =15` and `maxv = 4`. This uses a linear rescaling to force text (with total score < 15) to the range [-4,4], useful for comparing with manually tagged data.
Also `alpha` and `maxv` are now parameters of the `Nvader` object, so they can be changed easily!

**3. score breakdown**

It is sometimes useful to see what the word scores that lead to the final polarity were. Call
```
polarity_dict, tagged_sentence = Nvader.polarity_scores(s, return_tagged=True)
```
`polarity_dict`is the usual output while `tagged_sentence` is an `OrderedDict` of the sentence components and their polarity scores.

**4. Bootstrapping**

This is **the** major change. Usually the word scores in the VADER dictionary are obtained from averaging the scores of a number of raters. The individual scores are given in the 
`vader_lexicon.txt` file, we can create a team of synthethic raters by resampling `Nboots` scores from these lists of ratings. By propagating word score *vectors* instead of floats we get a 
*distribution* of possible polarity scores.
```
Nvader = NSentimentIntensityAnalyzer(Nboots=100)
s = "Nvader is not very smart." 
pol = Nvader.polarity_scores(s)
```
The element `pol['compound']` is a numpy array of size 100.

**5. Classification**

Often you just want to know if a sentence is positive, negative or neutral. There's now a function that does this by utilising the bootstrap.
```
Nvader.classify(text, conf=0.95, neutral_range=(0,0), method="majority_vote" ):
```

There are 2 methods
 - `'majority_vote'`: polarities `p` are considered neutral if `neutral_range[0] <= p <= neutral_range[1]`, positive if `p > neutral_range[1]` and negative if `p < neutral_range[0]`.
The counts of positive, negative and neutral are summed without regard to magnitude and the largest one is reported. In the case of a tie, the magnitudes are compared.
 - `'confidence'`: The polarities are sorted and the middle `conf` values are extracted. If they are all positive, negative or in the neutral range this is returned, otherwise `None` is returned






















