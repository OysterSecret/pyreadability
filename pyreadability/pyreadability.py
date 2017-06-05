import re
import string
import pkg_resources
import os
import nltk

DATA_PATH = pkg_resources.resource_filename('pyreadability', 'data/')

class SyllableCounter(object):
    def __init__(self):
        raise NotImplemented("class is an interface")

    def count_syllables(self, word):
        raise NotImplemented("class is an interface")

class SimpleSyllableCounter(SyllableCounter):
    def __init__(self):
        pass

    def count_syllables(self, word):
        count = 0
        vowels = "aeiou"

        lc_word = word.lower()

        last_c = " "
        for c in lc_word:
            if c in vowels and last_c not in vowels:
                count += 1
            last_c = c
        if lc_word.endswith('e'):
            count -= 1
        if lc_word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        count = max(count, 1)
        return int(count)

class CMUDictCounter(SyllableCounter):
    def __init__(self):
        from nltk.corpus import cmudict
        self.cmu_dict = cmudict.dict()
        self.fallback = SimpleSyllableCounter()

    def count_syllables(self, word):
        syllables = 0
        if word in self.cmu_dict:
            ph_list = self.cmu_dict[word][0]
            for ph in ph_list:
                has_number = any([c.isdigit() for c in ph])
                if has_number:
                    syllables+=1
        else:
            syllables = self.fallback.count_syllables(word)

        return max(syllables, 1)


class SentenceTokenizer(object):
    def __init__(self):
        raise NotImplemented("class is an interface")

    def tokenize(self, text):
        raise NotImplemented("class is an interface")

class SimpleSentenceTokenizer(SentenceTokenizer):
    def __init__(self):
        pass

    def tokenize(self, text):
        return re.split("[.!?]+", text)

class WordTokenizer(object):
    def __init__(self):
        raise NotImplemented("class is an interface")

    def tokenize(self, sentence):
        raise NotImplemented("class is an interface")

class SimpleWordTokenizer(WordTokenizer):
    def __init__(self):
        pass

    def tokenize(self, sentence):
        return sentence.split()

class NLTKSentenceTokenizer(SentenceTokenizer):
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.sent_tokenize(text)

class NLTKWordTokenizer(WordTokenizer):
    def __init__(self):
        pass

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence)


class Readability(object):
    _punc = string.punctuation
    _ttable = string.maketrans("", "")

    def __init__(self, syllable_counter = SimpleSyllableCounter(),
            sentence_tokenizer = SimpleSentenceTokenizer(), 
            word_tokenizer = SimpleWordTokenizer()):
        assert isinstance(syllable_counter, SyllableCounter)
        assert isinstance(sentence_tokenizer, SentenceTokenizer)
        assert isinstance(word_tokenizer, WordTokenizer)

        self.syllable_counter = syllable_counter
        self.sentence_tokenizer = sentence_tokenizer
        self.word_tokenizer = word_tokenizer

        self._load_dale_chall_difficult_words_3000()

    def _load_dale_chall_difficult_words_3000(self):
        fin = open(os.path.join(DATA_PATH,"dale_chall_3000.txt"), 'r')
        all_words = fin.read()
        words = all_words.split()
        self.dale_chall_words_3000 = set(words)


    def _count_difficult_words(self, tok_text):
        difficult_word_count = 0
        for sentence in tok_text:
            for word in sentence:
                if word in self.dale_chall_words_3000:
                    difficult_word_count += 1
        return difficult_word_count

    def _char_in_word_count(self, tok_text):
        char_count = 0
        for sentence in tok_text:
            for word in sentence:
                char_count += len(word)
        return char_count

    def _word_count(self, tok_text):
        word_count = 0
        for sentence in tok_text:
            word_count += len(sentence)
        return word_count

    def _sentence_count(self, tok_text):
        return len(tok_text)

    def _syllable_count(self, tok_text):
        syllable_count = 0
        for sentence in tok_text:
            for word in sentence:
                syllable_count += self.syllable_counter.count_syllables(word)

        return syllable_count

    def _count_average_letters_per_word(self, tok_text):
        total_chars = self._char_in_word_count(tok_text)
        word_count = self._word_count(tok_text)

        avg_letters_per_word = total_chars / float(word_count)

        return avg_letters_per_word

    def _count_average_words_per_sentence(self, tok_text):
        total_sentences = self._sentence_count(tok_text)
        word_count = self._word_count(tok_text)

        avg_words_per_sentence = word_count / float(total_sentences)
        
        return avg_words_per_sentence


    def _filter_word(self, word, lowercase, remove_punctuation, remove_digits):
        out_word = word
        
        if lowercase:
            out_word = word.lower()
        
        if remove_punctuation:
            out_word = out_word.translate(self._ttable, self._punc)

        if remove_digits:
            out_word = out_word.translate(self._ttable, string.digits)

        return out_word

    def _tokenize_text(self, text, lowercase=True, remove_puntuation=True, remove_digits=True):
        tok_text = []
        for sentence in self.sentence_tokenizer.tokenize(text):
            if not sentence:
                continue
            tok_sentence = []
            for word in self.word_tokenizer.tokenize(sentence):
                filtered_word = self._filter_word(word, lowercase, remove_puntuation, remove_digits)
                if not filtered_word:
                    continue
                tok_sentence.append(filtered_word)
            if not tok_sentence:
                continue
            tok_text.append(tok_sentence)
        return tok_text


    def flesch_kincaid_reading_ease(self, text):
        tok_text = self._tokenize_text(text)
        word_count = float(self._word_count(tok_text))
        sentence_count = float(self._sentence_count(tok_text))
        syllable_count = float(self._syllable_count(tok_text))

        fkre = 206.835 - 1.015 * (word_count / sentence_count) - \
                84.6 * (syllable_count / word_count)

        return fkre 
        
    def flesch_kincaid_grade_level(self, text):
        tok_text = self._tokenize_text(text)
        word_count = float(self._word_count(tok_text))
        sentence_count = float(self._sentence_count(tok_text))
        syllable_count = float(self._syllable_count(tok_text))

        fkgl = 0.39 * (word_count / sentence_count) + \
                11.8 * (syllable_count / word_count) - 15.59

        return fkgl

    def automated_readability_index(self, text):
        tok_text = self._tokenize_text(text)
        char_count = float(self._char_in_word_count(tok_text))
        word_count = float(self._word_count(tok_text))
        sentence_count = float(self._sentence_count(tok_text))
        ari = 4.71 * (char_count / word_count) + \
                0.5 * (word_count / sentence_count) - 21.43

        return ari

    def dale_chall_readability(self, text):
        tok_text = self._tokenize_text(text)
        difficult_word_count = float(self._count_difficult_words(tok_text))
        word_count = float(self._word_count(tok_text))
        sentence_count = float(self._sentence_count(tok_text))

        dcr = 0.1579 * ((difficult_word_count / word_count) * 100.0) +\
                0.0496 * (word_count / sentence_count)
        return dcr

    def coleman_liau_index(self, text):
        tok_text = self._tokenize_text(text)

        avg_words_per_sentence = self._count_average_words_per_sentence(tok_text)
        avg_letters_per_word = self._count_average_letters_per_word(tok_text)
        L = avg_letters_per_word * 100.0
        S = 100.0 / avg_words_per_sentence
        cli = 0.0588 * L - 0.296 * S - 15.8
        return cli
