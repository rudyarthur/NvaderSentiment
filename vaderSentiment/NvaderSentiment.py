# coding: utf-8
# Author: C.J. Hutto
# Thanks to George Berry for reducing the time complexity from something like O(N^4) to O(N).
# Thanks to Ewan Klein and Pierpaolo Pantone for bringing VADER into NLTK. Those modifications were awesome.
# For license information, see LICENSE.TXT

"""
If you use the VADER sentiment analysis tools, please cite:
Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

Modified by Rudy Arthur 2023 to use resampling and make some the rules optional
"""
import os
import re
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open

import csv
import numpy as np
from collections import OrderedDict

# ##Constants##

# (empirically derived mean sentiment intensity rating increase for booster words)
B_INCR = 0.293
B_DECR = -0.293
B_DECAY = 0.05 #Rudy: reduce effect of "really" as you get further away from it

# (empirically derived mean sentiment intensity rating increase for using ALLCAPs to emphasize a word)
C_INCR = 0.733
N_SCALAR = -0.74
NV_SCALAR = 1.25  #Rudy: another negation magic number

NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

# booster/dampener 'intensifiers' or 'degree adverbs'
# http://en.wiktionary.org/wiki/Category:English_degree_adverbs

BOOSTER_DICT = \
    {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR,
     "completely": B_INCR, "considerable": B_INCR, "considerably": B_INCR,
     "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormous": B_INCR, "enormously": B_INCR,
     "entirely": B_INCR, "especially": B_INCR, "exceptional": B_INCR, "exceptionally": B_INCR,
     "extreme": B_INCR, "extremely": B_INCR,
     "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR, "frackin": B_INCR, "fracking": B_INCR,
     "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR,
     "fuckin": B_INCR, "fucking": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,
     "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR,
     "incredible": B_INCR, "incredibly": B_INCR, "intensely": B_INCR,
     "major": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
     "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
     "so": B_INCR, "substantially": B_INCR,
     "thoroughly": B_INCR, "total": B_INCR, "totally": B_INCR, "tremendous": B_INCR, "tremendously": B_INCR,
     "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utter": B_INCR, "utterly": B_INCR,
     "very": B_INCR,
     "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
     "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
     "less": B_DECR, "little": B_DECR, "marginal": B_DECR, "marginally": B_DECR,
     "occasional": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
     "scarce": B_DECR, "scarcely": B_DECR, "slight": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
     "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# check for sentiment laden idioms that do not contain lexicon words (future work, not yet implemented)
SENTIMENT_LADEN_IDIOMS = {"cut the mustard": 2, "hand to mouth": -2,
                          "back handed": -2, "blow smoke": -2, "blowing smoke": -2,
                          "upper hand": 1, "break a leg": 2,
                          "cooking with gas": 2, "in the black": 2, "in the red": -2,
                          "on the ball": 2, "under the weather": -2}

# check for special case idioms and phrases containing lexicon words
SPECIAL_CASES = {"the shit": 3, "the bomb": 3, "bad ass": 1.5, "badass": 1.5, "bus stop": 0.0,
                 "yeah right": -2, "kiss of death": -1.5, "to die for": 3,
                 "beating heart": 3.1, "broken heart": -2.9 }


# #Static methods# #

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    input_words = [str(w).lower() for w in input_words]
    #neg_words = []
    #neg_words.extend(NEGATE) #Rudy: What's the point of this?
    #for word in neg_words:

    for word in NEGATE:
        if word in input_words:
            return True
    if include_nt:
        for word in input_words:
            if "n't" in word:
                return True
    #if "least" in input_words:
    #   i = input_words.index("least")
    #   if i > 0 and input_words[i - 1] != "at":
    #       return True
    return False



    
def allcap_differential(words):
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0
    for word in words:
        if word.isupper():
            allcap_words += 1
    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True
    return is_different





class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')
        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from\
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = allcap_differential(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token):
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)
        if len(stripped) <= 2:
            return token
        return stripped

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        stripped = list(map(self._strip_punc_if_word, wes))
        return stripped

class NSentimentIntensityAnalyzer(object):
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt", emoji_lexicon="emoji_utf8_lexicon.txt", Nboots=1, normtype='standard'):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)
        
        self.alpha = 15
        self.maxv = 4
        self.Nboots = Nboots
        self.normtype = normtype
        self.apply_boost = True
        self.apply_allcap = True
        self.apply_negation = True
        self.apply_punctemph = True
        self.apply_but = True
        self.apply_specialcases = True
        
        with codecs.open(lexicon_full_filepath, encoding='utf-8') as f:
            self.lexicon_full_filepath = f.read()
        self.lexicon = self.make_lex_dict()

        emoji_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), emoji_lexicon)
        with codecs.open(emoji_full_filepath, encoding='utf-8') as f:
            self.emoji_full_filepath = f.read()
        self.emojis = self.make_emoji_dict()
    
    def no_modify():
        self.apply_boost = False
        self.apply_allcap = False
        self.apply_punctemph = False
        self.apply_negation = False
        self.apply_but = False
        self.apply_specialcases = False
  
    def only_negate():
        self.apply_negation = True
        self.apply_boost = False
        self.apply_allcap = False
        self.apply_punctemph = False
        self.apply_but = False
        self.apply_specialcases = False

    def normalize(self, score):
        """
        Normalize the score to be between -1 and 1 using an alpha that
        approximates the max expected value
		
		Rudy: Differences between really positive or negative scores are squashed a lot
		Rudy: e.g. score = 2 -> 0.46, score = 4 -> 0.72. score = 15 -> 0.97, score = 30 -> 0.99
		Rudy: If you run this on longer texts, you'd want to increase alpha a lot!
		"""
		#norm_score = score / math.sqrt((score * score) + alpha)
		#if norm_score < -1.0: #Rudy: I don't think this is possible
		#    return -1.0
		#elif norm_score > 1.0:
		#    return 1.0
		#else:
		#    return norm_score
        return score / np.sqrt((score * score) + self.alpha)

    def linearnormalize(self, score):
        return self.maxv * score / self.alpha
    
    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        
        Rudy: if Nboot = 1 use values in col 1, otherwise randomly sample Nboots scores from the last column
        """
        lex_dict = {}
        
        #for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
        #   if not line:
        #        continue
        #    (word, measure) = line.strip().split('\t')[0:2]
        #    lex_dict[word] = float(measure)
        csvreader = csv.reader(self.lexicon_full_filepath.rstrip('\n').split('\n'), delimiter='\t')
        for row in csvreader:
            w = row[0]
            if self.Nboots == 1:
                x = float(row[1])
            else: 
                vals = np.array([ float(v) for v in row[-1][1:][:-1].split(",") ])				
                if self.Nboots <= len(vals):
                    x = vals[:self.Nboots]
                else:
                    x = np.random.choice( vals, self.Nboots )
            lex_dict[w] = x
        return lex_dict

    def make_emoji_dict(self):
        """
        Convert emoji lexicon file to a dictionary
        """
        emoji_dict = {}
        for line in self.emoji_full_filepath.rstrip('\n').split('\n'):
            (emoji, description) = line.strip().split('\t')[0:2]
            emoji_dict[emoji] = description
        return emoji_dict

    def classify(self, text, conf=0.95, neutral_range=(0,0), method="majority_vote" ):
        """
        Turn a score array into a classification pos/neg/neutral None = unclassified
        majority_vote method always returns a label
        confidence method can return None
        Just does a simple comparison for Nboots=1
        """
        score_array = self.polarity_scores(text)['compound']
        if self.Nboots == 1: 
            if score_array >= neutral_range[0] and score_array <= neutral_range[1]: return 0
            if score_array > neutral_range[1]: return 1
            return -1
        
        if method == "median" or method == "mean":
			
            if method == "median": score = np.median(score_array)
            if method == "mean": score = np.mean(score_array)
            
            if score >= neutral_range[0] and score <= neutral_range[1]: return 'neu'
            if score > neutral_range[1]: return 'pos'
            return 'neg'
			
        if method == "confidence":
            rs = np.sort(score_array)
            lb, ub = rs[ round( 0.5*(1-conf)*self.Nboots) ], rs[ round( 0.5*(1+conf)*self.Nboots)-1 ]    
            if lb >= neutral_range[0] and ub <= neutral_range[1]: return 'neu'
            if lb > neutral_range[1] and ub > neutral_range[1]: return 'pos'
            if lb < neutral_range[0] and ub < neutral_range[0]: return 'neg'
            return None
            
            
        #if method == "majority_vote":
        cneu = np.count_nonzero( (score_array >= neutral_range[0]) * (score_array <= neutral_range[1])	)		
        cneg = np.count_nonzero( (score_array < neutral_range[0]) 	)		
        cpos = np.count_nonzero( (score_array > neutral_range[1]) 	)		
		
        cmax = max(cneu, max(cneg, cpos))
		#tiebreaker - chose the one with the greatest sum
        if cmax == cpos and cpos == cneg:
            sneg = np.abs(np.where( (score_array < neutral_range[0]), score_array, 0 ).sum())
            spos = np.where( (score_array > neutral_range[1]), score_array, 0 ).sum()
            if spos > sneg: return 'pos'
            return 'neg'
        #to choose between neutral and positive sum distance from extremes 
        if cmax == cpos and cpos == cneu:
            neu_centre = (neutral_range[1] + neutral_range[0])*0.5
            sneu = np.abs(np.where( (score_array >= neutral_range[0]) * (score_array <= neutral_range[1]), score_array-neu_centre, 0 ).sum())
            spos = np.abs(np.where( ( score_array > neutral_range[1]), score_array-1.0, 0 ).sum())
            if spos <= sneu: return 'pos'
            return 'neu'
        if cmax == cneg and cneg == cneu:	
            neu_centre = (neutral_range[1] + neutral_range[0])*0.5
            sneu = np.abs(np.where( (score_array >= neutral_range[0]) * (score_array <= neutral_range[1]), score_array-neu_centre, 0 ).sum())
            sneg = np.abs(np.where( ( score_array < neutral_range[0]), score_array+1, 0 ).sum())
            if sneg <= sneu: return 'neg'
            return 'neu'
            
        if cmax == cpos: return 'pos'
        if cmax == cneg: return 'neg'
        if cmax == cneu: return 'neu'
        
        return None
		
				
    def polarity_scores(self, text, return_tagged=False):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # convert emojis to their textual descriptions
        text_no_emoji = ""
        prev_space = True
        for c in text:
            if c in self.emojis:
                # get the textual description
                description = self.emojis[c]
                if not prev_space:
                    text_no_emoji += ' '
                text_no_emoji += description
                prev_space = False
            else:
                text_no_emoji += c
                prev_space = c == ' '
        text = text_no_emoji.strip()

        sentitext = SentiText(text)

        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons

        for i, item in enumerate(words_and_emoticons):
            valence = 0
            # check for vader_lexicon words that may be used as modifiers or negations
            if self.apply_boost and item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue
            if self.apply_boost and (i < len(words_and_emoticons) - 1 and item.lower() == "kind" and
                    words_and_emoticons[i + 1].lower() == "of"):
                sentiments.append(valence)
                continue
            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)
            
        if self.apply_but: sentiments = self._but_check(words_and_emoticons, sentiments) 
        if all([ not isinstance(x,np.ndarray) for x in sentiments]): sentiments[0] = np.zeros(self.Nboots)
        valence_dict = self.score_valence(sentiments, text)
		
        if return_tagged: return valence_dict, self.tagged_text(sentiments, words_and_emoticons)
        return valence_dict

    def tagged_text(self, sentiments, words_and_emoticons):
        tagged = OrderedDict()
        for i, item in enumerate(words_and_emoticons): tagged[item] = sentiments[i]
        return tagged
		
    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = self.apply_allcap and sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence 
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for an adjacent lexicon item vs "no" as its own stand-alone lexicon item
            if item_lowercase == "no" and i != len(words_and_emoticons)-1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0
            if (i > 0 and words_and_emoticons[i - 1].lower() == "no") \
               or (i > 1 and words_and_emoticons[i - 2].lower() == "no") \
               or (i > 2 and words_and_emoticons[i - 3].lower() == "no" and words_and_emoticons[i - 1].lower() in ["or", "nor"] ):
                valence = self.lexicon[item_lowercase] * N_SCALAR

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if self.Nboots == 1:
                    valence += np.sign(valence) * C_INCR
                else:
                    valence += np.where(valence > 0, C_INCR, -C_INCR)

                #if valence > 0:
                #    valence += C_INCR
                #else:
                #    valence -= C_INCR

			
            for start_i in range(0, 3):
                # dampen the scalar modifier of preceding words and emoticons
                # (excluding the ones that immediately preceed the item) based
                # on their distance from the current item.
                if i > start_i and words_and_emoticons[i - (start_i + 1)].lower() not in self.lexicon:
                    if self.apply_boost:
                        s = self.scalar_inc_dec(words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff)
                        
                        if start_i: s *= (1.-start_i*B_DECAY)
						  							
                        #if start_i == 1 and s != 0:
                        #    s = s * 0.95
                        #if start_i == 2 and s != 0:
                        #    s = s * 0.9
                        valence = valence + s


                    if self.apply_negation: valence = self._negation_check(valence, words_and_emoticons, start_i, i)
                    if self.apply_specialcases and start_i == 2: valence = self._special_idioms_check(valence, words_and_emoticons, i)  

            valence = self._least_check(valence, words_and_emoticons, i) 

        sentiments.append(valence)
        return sentiments

    def scalar_inc_dec(self, word, valence, is_cap_diff):
        """
        Check if the preceding words increase, decrease, or negate/nullify the
        valence
        """
        scalar = 0.0
        word_lower = word.lower()
        if word_lower in BOOSTER_DICT:
            scalar = BOOSTER_DICT[word_lower]
        
            if self.Nboots == 1:
                if valence < 0: scalar *= -1
            else:
                scalar = np.where(valence > 0, scalar, -scalar)
                				
            # check if booster/dampener word is in ALLCAPS (while others aren't)
            if self.apply_allcap and word.isupper() and is_cap_diff:
                #if valence > 0:
                #    scalar += C_INCR
                #else:
                #    scalar -= C_INCR
                if self.Nboots == 1:
                    scalar += (1 - 2*(valence < 0))*C_INCR
                else:
                    scalar += np.where(valence > 0, C_INCR, -C_INCR)
                
        return scalar
    
    def _least_check(self, valence, words_and_emoticons, i):
        #Rudy: Could check after as well: least like -vs- like least?
        # check for negation case using "least"
        if i > 1 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            if words_and_emoticons[i - 2].lower() != "at" and words_and_emoticons[i - 2].lower() != "very":
                valence = valence * N_SCALAR
        elif i > 0 and words_and_emoticons[i - 1].lower() not in self.lexicon \
                and words_and_emoticons[i - 1].lower() == "least":
            valence = valence * N_SCALAR
        return valence

    def _but_check(self, words_and_emoticons, sentiments):
        # check for modification in sentiment due to contrastive conjunction 'but'
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if 'but' in words_and_emoticons_lower:
            bi = words_and_emoticons_lower.index('but')

			
			#Rudy: very weird, is there some reason it's like this e.g. if there are duplicate values si won't be the correct index?
            #for sentiment in sentiments:
            #    si = sentiments.index(sentiment)
            #    if si < bi:
            #        sentiments.pop(si)
            #        sentiments.insert(si, sentiment * 0.5)
            #    elif si > bi:
            #        sentiments.pop(si)
            #        sentiments.insert(si, sentiment * 1.5)
            for si,sentiment in enumerate(sentiments):
                if si < bi:
                    sentiments[si] *= 0.5 
                elif si > bi:
                    sentiments[si] *= 1.5

        return sentiments

    def _special_idioms_check(self, valence, words_and_emoticons, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        onezero = "{0} {1}".format(words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoonezero = "{0} {1} {2}".format(words_and_emoticons_lower[i - 2],
                                          words_and_emoticons_lower[i - 1], words_and_emoticons_lower[i])

        twoone = "{0} {1}".format(words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwoone = "{0} {1} {2}".format(words_and_emoticons_lower[i - 3],
                                           words_and_emoticons_lower[i - 2], words_and_emoticons_lower[i - 1])

        threetwo = "{0} {1}".format(words_and_emoticons_lower[i - 3], words_and_emoticons_lower[i - 2])

        sequences = [onezero, twoonezero, twoone, threetwoone, threetwo]

        for seq in sequences:
            if seq in SPECIAL_CASES:
                valence = SPECIAL_CASES[seq]
                if self.Nboots > 1: valence = valence*np.ones(self.Nboots)
                break

        if len(words_and_emoticons_lower) - 1 > i:
            zeroone = "{0} {1}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1])
            if zeroone in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroone]
                if self.Nboots > 1: valence = valence*np.ones(self.Nboots)

        if len(words_and_emoticons_lower) - 1 > i + 1:
            zeroonetwo = "{0} {1} {2}".format(words_and_emoticons_lower[i], words_and_emoticons_lower[i + 1],
                                              words_and_emoticons_lower[i + 2])
            if zeroonetwo in SPECIAL_CASES:
                valence = SPECIAL_CASES[zeroonetwo]
                if self.Nboots > 1: valence = valence*np.ones(self.Nboots)

        # check for booster/dampener bi-grams such as 'sort of' or 'kind of'
        n_grams = [threetwoone, threetwo, twoone]
        for n_gram in n_grams:
            if n_gram in BOOSTER_DICT:
                valence = valence + BOOSTER_DICT[n_gram]

        return valence

    @staticmethod
    def _sentiment_laden_idioms_check(valence, senti_text_lower): ##Rudy: Is not called anywhere
        # Future Work
        # check for sentiment laden idioms that don't contain a lexicon word
        idioms_valences = []
        for idiom in SENTIMENT_LADEN_IDIOMS:
            if idiom in senti_text_lower:
                valence = SENTIMENT_LADEN_IDIOMS[idiom]
                idioms_valences.append(valence)
        if len(idioms_valences) > 0:
            valence = sum(idioms_valences) / float(len(idioms_valences))
        return valence

    @staticmethod
    def _negation_check(valence, words_and_emoticons, start_i, i):
        words_and_emoticons_lower = [str(w).lower() for w in words_and_emoticons]
        if start_i == 0:
            if negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 1 word preceding lexicon word (w/o stopwords)
                valence = valence * N_SCALAR
        if start_i == 1:
            if words_and_emoticons_lower[i - 2] == "never" and \
                    (words_and_emoticons_lower[i - 1] == "so" or
                     words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * NV_SCALAR
            elif words_and_emoticons_lower[i - 2] == "without" and \
                    words_and_emoticons_lower[i - 1] == "doubt":
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 2 words preceding the lexicon word position
                valence = valence * N_SCALAR
        if start_i == 2:
            if words_and_emoticons_lower[i - 3] == "never" and \
                    (words_and_emoticons_lower[i - 2] == "so" or words_and_emoticons_lower[i - 2] == "this") or \
                    (words_and_emoticons_lower[i - 1] == "so" or words_and_emoticons_lower[i - 1] == "this"):
                valence = valence * NV_SCALAR
            elif words_and_emoticons_lower[i - 3] == "without" and \
                    (words_and_emoticons_lower[i - 2] == "doubt" or words_and_emoticons_lower[i - 1] == "doubt"):
                valence = valence
            elif negated([words_and_emoticons_lower[i - (start_i + 1)]]):  # 3 words preceding the lexicon word position
                valence = valence * N_SCALAR
        return valence

    def _punctuation_emphasis(self, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    @staticmethod
    def _amplify_ep(text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    @staticmethod
    def _amplify_qm(text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    def _sift_sentiment_scores(self, sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = np.zeros(self.Nboots)
        neg_sum = np.zeros(self.Nboots)
        neu_count = np.zeros(self.Nboots)
        for sentiment_score in sentiments:
            sentiment_score = np.array(sentiment_score) #Rudy: save some writing
            pos_sum += np.where(sentiment_score > 0, sentiment_score+1, 0)		
            neg_sum += np.where(sentiment_score < 0, sentiment_score-1, 0)		
            neu_count += np.where(sentiment_score == 0, 1, 0)		
            #if sentiment_score > 0:
            #    pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1 Rudy: the 1s are weird
            #if sentiment_score < 0:
            #    neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            #if sentiment_score == 0:
            #    neu_count += 1
        if self.Nboots == 1: return pos_sum[0], neg_sum[0], neu_count[0]
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
			
            
            if self.Nboots == 1:
                sum_s = float(sum(sentiments))
            else:
                sum_s = np.stack( [s for s in sentiments if isinstance(s, np.ndarray) ], axis=1).sum(axis=1)
                
            # compute and add emphasis from punctuation in text 
            if self.apply_punctemph:
                punct_emph_amplifier = self._punctuation_emphasis(text)
                if self.Nboots == 1:
                    sum_s += np.sign(sum_s)*punct_emph_amplifier
                else:
                    sum_s += np.where( sum_s > 0, punct_emph_amplifier, -punct_emph_amplifier) 					
                #if sum_s > 0:
                #    sum_s += punct_emph_amplifier
                #elif sum_s < 0:
                #    sum_s -= punct_emph_amplifier
            if self.normtype == "raw": 
                compound = sum_s
            elif self.normtype == "linear":
                compound = self.linearnormalize(sum_s)
            else:
                compound = self.normalize(sum_s)   				
            

            # discriminate between positive, negative and neutral sentiment scores TODO
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if self.apply_punctemph:
                punct_emph_amplifier = self._punctuation_emphasis(text)
                if self.Nboots == 1:
                    pos_sum += (pos_sum > math.fabs(neg_sum))*punct_emph_amplifier
                    neg_sum -= (pos_sum < math.fabs(neg_sum))*punct_emph_amplifier
                else:
                    pos_sum += np.where( pos_sum > np.abs(neg_sum), punct_emph_amplifier, 0) 	            
                    neg_sum += np.where( pos_sum < np.abs(neg_sum), -punct_emph_amplifier, 0) 	            
            #    if pos_sum > math.fabs(neg_sum):
            #        pos_sum += punct_emph_amplifier
            #    elif pos_sum < math.fabs(neg_sum):
            #        neg_sum -= punct_emph_amplifier

            total = pos_sum + np.abs(neg_sum) + neu_count
            pos = np.abs(pos_sum / total)
            neg = np.abs(neg_sum / total)
            neu = np.abs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
            {"neg": neg,
             "neu": neu,
             "pos": pos,
             "compound": compound}

        return sentiment_dict


if __name__ == '__main__':
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    
	
    sentences = []
    with open("../additional_resources/hutto_ICWSM_2014/tweets_GroundTruth.txt", 'r') as infile:
	    csvreader = csv.reader(infile, delimiter="\t")
	    for row in csvreader: sentences.append(  row[-1]	)

    sentiment_dist = []
    with open("../additional_resources/hutto_ICWSM_2014/tweets_anonDataRatings.txt", 'r') as infile:
        csvreader = csv.reader(infile, delimiter="\t")
        for row in csvreader: sentiment_dist.append(  np.array([ float(v) for v in row[-1][1:][:-1].split(",") ])	)

    neg_test = [0, 0]
    neu_test = [0, 0]
    pos_test = [0, 0]
    hard_test = [np.inf, 0]

    for idx,d in enumerate(sentiment_dist):
	    negsum = np.where( d<0, 1, 0 ).sum()
	    possum = np.where( d>0, 1, 0 ).sum()
	    neusum = np.where( d==0, 1, 0 ).sum()
	    if negsum > neg_test[0]: neg_test = [negsum, idx]
	    if possum > pos_test[0]: pos_test = [possum, idx]
	    if neusum > neu_test[0]: neu_test = [neusum, idx]
	    if possum > neusum and negsum > neusum and negsum+possum < hard_test[0]: hard_test = [negsum+possum, idx]

    fig, ax = plt.subplots(2,2, figsize=(20,10) )
    analyzer = NSentimentIntensityAnalyzer(Nboots=100, normtype="linear")	
    vader = NSentimentIntensityAnalyzer(Nboots=1, normtype="linear")
    
    def ax_plot(ax, idx):
        s = sentences[idx]
        print(s)
        d = sentiment_dist[idx]
        vpol = vader.polarity_scores(s)
        npol = analyzer.polarity_scores(s)
        ax.hist(d, bins=np.arange(-4,5,1), zorder=1, density=True)	
        ax.axvline(x=np.mean(d), linestyle="-", color='r', zorder=2, linewidth=3, label="Mean:{:.3f}".format(np.mean(d)))
        ax.axvline(x=vpol['compound'], linestyle="--", color='k', zorder=2, label="VADER:{:.3f}".format(vpol['compound']))
        ax.hist(npol['compound'], bins=np.arange(-4,5,1), label = "NVADER", histtype="step", zorder=3, density=True)
        if len(s.split()) > 10: 
            ax.set_title( " ".join(s.split()[:10]) + "\n" + " ".join(s.split()[10:]) )
        else: 
            ax.set_title(s)
        ax.legend(loc="upper left")
		
    ax_plot(ax[0][0], neg_test[1])
    ax_plot(ax[0][1], pos_test[1])
    ax_plot(ax[1][0], neu_test[1])
    ax_plot(ax[1][1], hard_test[1])
    plt.savefig("output.png", dpi=fig.dpi)
    plt.show()		
	
	
