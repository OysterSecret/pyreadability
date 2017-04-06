import numpy as np
from pyreadability.pyreadability import *

def test_flesch_reading_ease():
    text = "This is a simple test text"
    rb = Readability()
    print rb.flesch_kincaid_reading_ease(text)

def test_flesch_grade_level():
    text = "This is a simple test text readability"
    rb = Readability()
    print rb.flesch_kincaid_grade_level(text)

def test_flesch_grade_level_monosyllable():
    text = "The. Cat. And. The. Dog. Ate."
    rb = Readability()
    assert np.isclose(rb.flesch_kincaid_grade_level(text),-3.4)

def test_automated_readability_index():
    text = "This is a simple test text readability"
    rb = Readability()
    print rb.automated_readability_index(text)

def test_coleman_liau_index():
    text = """Existing computer programs that measure readability are based largely upon subroutines which estimate number of syllables, usually by counting vowels. The shortcoming in estimating syllables is that it necessitates keypunching the prose into the computer. There is no need to estimate syllables since word length in letters is a better predictor of readability than word length in syllables. Therefore, a new readability formula was computed that has for its predictors letters per 100 words and sentences per 100 words. Both predictors can be counted by an optical scanning device, and thus the formula makes it economically feasible for an organization such as the U.S. Office of Education to calibrate the readability of all textbooks for the public school system."""
    rb = Readability()
    print rb.coleman_liau_index(text)

def test_flesch_reading_ease_wikipedia():
    text1 = "This sentence, taken as a reading passage unto itself, is being used to prove a point."
    rb = Readability(syllable_counter=CMUDictCounter())
    fkre = rb.flesch_kincaid_reading_ease(text1)
    #NB: wikipedia miscounts the number of syllables in the sentence. This should be correct.
    assert np.isclose(fkre, 68.9, atol=0.1, rtol=0.1)

    text2 = "The Australian platypus is seemingly a hybrid of a mammal and reptilian creature."
    fkre = rb.flesch_kincaid_reading_ease(text2)
    assert np.isclose(fkre, 37.5, atol=0.1, rtol=0.1)

    text3 = text1 + " " + text1
    fkre = rb.flesch_kincaid_reading_ease(text3)
    assert np.isclose(fkre, 68.9, atol=0.1, rtol=0.1)
