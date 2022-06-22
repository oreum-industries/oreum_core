# curate.text_clean.py
# copyright 2022 Oreum Industries
import itertools
import string
import sys

import ftfy
import numpy as np
import regex as re
import requests
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from orderedset import OrderedSet
from pybloomfilter import BloomFilter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class SnakeyLowercaser:
    """Clean and standardise a string to snakey_lowercase
    Split CamelCasedStrings
    Convert '-' and '.' to '_' and preserve existing '_',
    Useful for the often messy column names present in Excel tables
    """

    def __init__(self):
        punct_to_remove = re.sub(r'_', '', string.punctuation)
        self.rx_to_underscore = re.compile(r'[-/]')
        self.rx_punct = re.compile('[{}]'.format(re.escape(punct_to_remove)))
        self.rx_splitter1 = re.compile(r'([A-Za-z0-9])([A-Z][a-z]+)')
        self.rx_patsy_factor = re.compile(r'^(.*)(\[T\.|\[)(.*)(\])(.*)$')
        self.rx_patsy_numpy = re.compile(r'^np\.(.*)\((.*)\)$')
        self.rx_patsy_interaction = re.compile(r':')
        self.rx_multi_underscore = re.compile(r'[_]{2,}')

    def clean(self, s):
        s0 = self.rx_to_underscore.sub('_', str(s))
        s1 = self.rx_punct.sub('', s0)
        s2 = self.rx_splitter1.sub(r'\1_\2 ', s1)
        s3 = '_'.join(s2.lower().split())
        s4 = self.rx_multi_underscore.sub('_', s3)
        return s4

    def clean_patsy(self, s):
        s0 = str(s).replace('-', '_')
        if len(f := self.rx_patsy_factor.findall(s0)) > 0:
            s0 = f[0][0] + '_t_' + f[0][2] + f[0][4]
        # run twice as a very lazy alt to recursion to cover interaction of 2 cats
        if len(f := self.rx_patsy_factor.findall(s0)) > 0:
            s0 = f[0][0] + '_t_' + f[0][2] + f[0][4]
        if len(f := self.rx_patsy_numpy.findall(s0)) > 0:
            s0 = f[0][1] + '_' + f[0][0]
        s1 = self.rx_patsy_interaction.sub('_x_', s0)
        s2 = self.rx_punct.sub('', s1)
        s3 = self.rx_splitter1.sub(r'\1_\2 ', s2)
        s4 = '_'.join(s3.lower().split())
        s5 = self.rx_multi_underscore.sub('_', s4)
        return s5


class TextCleaner:
    def __init__(self):
        self.rx_line = re.compile(re.escape('=\n'))  # "=\n"
        self.rx_nbsp = re.compile(r'&nbsp;')  # nbsp
        self.rx_copy = re.compile(r'&copy;')  # nbsp
        self.rx_numbers = re.compile(
            r'(\b[0-9,.\-\\\/]+\b)'
        )  # number blocks, money amounts, years etc
        self.rx_punct = re.compile(
            r'[{}]+'.format(re.escape(string.punctuation))
        )  # regular punctuation
        self.rx_neg_apostrophe = re.compile(
            r"""\b(ca|do|wo|is|are|was|does|
                    shall|should|would|could|must|ai)(n)(?:\')(t)\b""",
            re.I,
        )  # find apsotrophes in negations
        self.rx_hex = re.compile(r'=[a-f0-9]{2}', re.I)  # stray hexadecimal
        self.rx_arrows = re.compile('(>)+')  # sequences of ">" at start of line
        self.rx_repeatgt3 = re.compile(r'(.)\1{3,}')  # any char repeating more than 3x
        self.rx_htmlcom = re.compile(
            re.escape('<!--') + "(.*?)" + re.escape('-->'), re.DOTALL
        )  # HTML comments (usually embedded CSS)
        self.rx_email = re.compile(
            r"""\b[a-z0-9\'._%+-]+@[a-z0-9.-]+
                                    \.[a-z]{2,4}\b""",
            re.I,
        )  # email addresses
        self.rx_web = re.compile(
            r"""((http[s]?\:\/\/)?(www\.)+
                                    [a-z0-9\-%\/.]+)""",
            re.I,
        )  # web addresses
        self.postcode = re.compile(
            r"""(GIR ?0AA|[A-PR-UWYZ]([0-9]{1,2}|([A-HK-Y][0-9]([0-9ABEHMNPRV-Y])?)
                |[0-9][A-HJKPS-UW])?[0-9][ABD-HJLNP-UW-Z]{2})""",
            re.I,
        )
        self.natins = re.compile(
            r'\s*[a-zA-Z]{2}(?:\s*\d\s*){6}[a-zA-Z]?\s*'
        )  # UK national insurance ID number
        self.phoneno = re.compile(
            r"""\(?(?:(?:0(?:0|11)\)?[\s-]?\(?|\+)44\)?[\s-]?\(?(?:0\)?[\s-]?
                \(?)?|0)(?:\d{2}\)?[\s-]?\d{4}[\s-]?\d{4}|\d{3}\)?[\s-]?\d{3}
                [\s-]?\d{3,4}|\d{4}\)?[\s-]?(?:\d{5}|\d{3}[\s-]?\d{3})|\d{5}\)?
                [\s-]?\d{4,5}|8(?:00[\s-]?11[\s-]?11|45[\s-]?46[\s-]?4\d))
                (?:(?:[\s-]?(?:x|ext\.?\s?|\#)\d+)?)"""
        )  # phone number
        self.nhs = re.compile(r'\d{3}\s?\d{3}\s?\d{4}')  # UK NHS ID number
        # self.rx_nonchar = re.compile(r'[\d{}]+'.format(
        #     re.escape(string.punctuation)))
        self.rx_num_m = re.compile(
            r'^(?P<mill>[0-9]+?(?:[.]+?[0-9]+?)*?)m$', re.I
        )  # (1.4)M
        self.rx_num_k = re.compile(
            r'^(?P<thou>[0-9]+?(?:[.]+?[0-9]+?)*?)k$', re.I
        )  # (400)k
        self.rx_num = re.compile(
            r'^(?P<whol>[0-9]+?)(?P<frac>[.]+?[0-9]+?)*?$', re.I
        )  # (81)(.23)
        self.rx_number_junk = re.compile(r'[#$€£₤¥,;%]')

    def fix_unicode(self, txt):
        """Fix bad unicode / emojis etc and try to remove crud"""

        t = ftfy.fix_text(txt, fix_character_width=False)  # fix encoding
        t = self.rx_hex.sub('', t)  # remove hex like '=b7', '=f5' etc

        return t

    def basic_clean(self, txt):
        """Clean up single raw text string where words have single spaces
        Note:
            doesnottokenise
            dOes Not chAnge Casing: allows proper noun removal later
        """
        t = self.rx_line.sub('', txt)
        t = self.rx_arrows.sub('', t)
        t = self.rx_repeatgt3.sub('', t)
        t = self.rx_nbsp.sub('', t)
        t = self.rx_htmlcom.sub('', t)
        t = self.rx_neg_apostrophe.sub('\1\2\3', t)
        t = self.rx_email.sub('', t)
        t = self.rx_web.sub('', t)
        t = self.rx_numbers.sub('', t)

        return t

    def convert_bad_number_representation_to_float(self, s):
        """Accept a string that represents a number in a shitty way and convert to float
        Issues:
                Currently limited to k and M.
                Hard to catch all usecases so returns nan on fail
                Yields % 100 too high
        Corrects a multitide of sins e.g:
            '1M' -> 1000000.
            '1.4m' -> 1400000.
            '25K' -> 25000.
            '81.12k' -> 81120.
            '3,000.82' -> 3000.82
            '$3,000.82' -> 3000.82
            '3.71%' -> 3.71

        ts = ['1M', '1.4m', '25K', '81.12k', '400', '9,000.23', '$35,000.82', '3.71%']
        for t in ts:
            print(t, convert_bad_number_representation_to_float(t))
        """
        r = np.nan
        s0 = self.rx_number_junk.sub('', s.strip().lower())
        gm = self.rx_num_m.match(s0)
        gk = self.rx_num_k.match(s0)
        gn = self.rx_num.match(s0)

        if gm is not None:
            mill = gm.capturesdict()['mill']
            mill = mill[0] if len(mill) > 0 else '0'
            r = np.float64(f'{mill}') * 1e6

        elif gk is not None:
            thou = gk.capturesdict()['thou']
            thou = thou[0] if len(thou) > 0 else '0'
            r = np.float64(f'{thou}') * 1e3

        elif gn is not None:
            whol = gn.capturesdict()['whol']
            whol = whol[0] if len(whol) > 0 else ''
            frac = gn.capturesdict()['frac']
            frac = frac[0] if len(frac) > 0 else '.0'
            r = np.float64(f'{whol}') + np.float64(f'{frac}')

        return r


class StopWorder:
    """Provide generalised stopwording using a bloomfilter"""

    def __init__(self):
        self.default_book_url = 'https://www.gutenberg.org/cache/epub/1661/pg1661.txt'
        self.names_first = 'data/stopwords/names_first.csv'
        self.names_last = 'data/stopwords/names_last.csv'
        self.global_cities = 'data/stopwords/cities1000_nameonly.csv'
        self.british_locations = 'data/stopwords/great-britain-latest_names_sorted.csv'
        self.unix_words = '/usr/share/dict/words'
        self.rx_nonchar = re.compile(
            r'[\d{}]+'.format(re.escape(string.punctuation))
        )  # punctuation or numbers
        self.lemtzr = WordNetLemmatizer()
        self.ng = NGrammer()

    def _get_book(self, book_url=None):
        """Download a book from project Gutenberg for testing"""

        if book_url is None:
            book_url = self.default_book_url

        try:
            r = requests.get(book_url)
            if r.ok:
                bookstart = 'START OF THIS PROJECT GUTENBERG EBOOK'
                bookend = 'END OF THIS PROJECT GUTENBERG EBOOK'
                txt_extract = r.text.partition(bookstart)[2].partition(bookend)[0]
                r.close()
                return txt_extract.replace('\r', '')

        except requests.ConnectionError as e:
            print('Cannot access URL: {}\n\n{}'.format(book_url, e))
        except requests.HTTPError as e:
            print('HTTP Error, perhaps server is down? \n\n{}'.format(e))
        except Exception as e:
            print(e)
        return 'no content'

    def _get_names(self):
        """Load a huge list of first and last names from csv"""

        with open(self.names_first, 'r') as f:
            fns = ['{}{}'.format(x[0].upper(), x.strip()[1:]) for x in f.readlines()][
                1:
            ]

        with open(self.names_last, 'r') as f:
            lns = ['{}{}'.format(x[0].upper(), x.strip()[1:]) for x in f.readlines()][
                1:
            ]

        f.close()
        print('Read in {} first names, and {} last names'.format(len(fns), len(lns)))
        return fns + lns

    def _get_cities1000(self):
        """Load a list of global cities with pop > 1000 from csv"""

        with open(self.global_cities, 'r') as f:
            cities = [
                '{}{}'.format(x[0].upper(), x.strip()[1:]) for x in f.readlines()
            ][1:]
        f.close()

        print('Read in {} global cities where population > 1000'.format(len(cities)))
        return cities

    def _get_britishlocations(self):
        """Load a big list of british locations from csv"""

        rx_apos = re.compile(r'\&apos\;')
        rx_quot = re.compile(r'\&quot\;')
        rx_amp = re.compile(r'\&amp\;')
        rx_num = re.compile(r'[0-9]{1,}')
        rx_chars = re.compile(r'^[A-Za-z]{1,}')

        def use_loc(s):
            s = s.strip()

            if rx_chars.match(s) is not None:
                if len(s) > 1:
                    return True

            return False

        def clean_locs(s):
            s = rx_apos.sub("'", s)
            s = rx_quot.sub('', s)
            s = rx_amp.sub('&', s)
            s = rx_num.sub('', s)

            return s.strip()

        with open(self.british_locations, 'r') as f:
            locs = [clean_locs(x) for x in f.readlines() if use_loc(x)][1:]

        f.close()
        print('Read in {} British locations'.format(len(locs)))
        return locs

    def _get_unix_words(self):
        """Get common words from Unix `words`.
        NOTE: This disk location is correct for Mac OSX 10.10, YMMV.
        """
        rx_isupper = re.compile('^[A-Z]{1,}')

        with open(self.unix_words, 'r') as f:
            wrds = [
                '{}'.format(x.strip().lower())
                for x in f.readlines()
                if rx_isupper is not None
            ]

        print('Read in {} common words from Unix `words`'.format(len(wrds)))
        return wrds

    def _get_email_filler(self):
        """Create a list of common chatter / filler words"""

        days = [
            'monday',
            'tuesday',
            'wednesday',
            'thursday',
            'friday',
            'saturday',
            'sunday',
        ]

        abv_days = [
            'mon',
            'tues',
            'tue',
            'weds',
            'wed',
            'thurs',
            'thur',
            'fri',
            'sat',
            'sun',
        ]

        months = [
            'january',
            'february',
            'march',
            'april',
            'may',
            'june',
            'july',
            'august',
            'september',
            'october',
            'november',
            'december',
        ]

        abv_months = [
            'jan',
            'feb',
            'mar',
            'apr',
            'may',
            'jun',
            'jul',
            'aug',
            'sept',
            'sep',
            'oct',
            'nov',
            'dec',
        ]

        email_cruft = [
            'from',
            'frm',
            'to',
            're',
            'fwd',
            'fw',
            'forward',
            'forwarded',
            'cc',
            'bcc',
            'sent',
            'subject',
            'subj',
            'reply-to',
            '<mime-attachment.gif>',
        ]

        email_cruft_full = email_cruft + ['{}:'.format(w) for w in email_cruft]

        html_cruft = ['font']

        contact_info = [
            'email',
            'e-mail',
            'e',
            'mail',
            'mobile',
            'mob',
            'm',
            'telephone',
            'phone',
            'tel',
            't',
            'fax',
            'http',
            'https',
            'www',
            'website',
            'web',
            'url',
            'address',
            'add',
            'switchboard',
            'mob',
            'tel',
            'web',
        ]

        sals = ['dear', 'hello', 'hi', 'hey', 'thanks for']

        vals = [
            'sincerely',
            'faithfully',
            'kind regards',
            'kindest regards',
            'best regards',
            'warm regards',
            'regards',
            'rgds',
            'thank you',
            'many thanks',
            'much thanks',
            'thanks',
            'cheers',
        ]

        yet_more = [
            'mr',
            'mrs',
            'ms',
            'attached',
            'ok',
            'okay',
            'ha',
            'wa',
            'date',
            'let',
            'message',
        ]

        filler = (
            days
            + abv_days
            + months
            + abv_months
            + email_cruft_full
            + html_cruft
            + contact_info
            + sals
            + vals
            + yet_more
        )

        print('Created {} email filler words '.format(len(filler)))
        return filler

    def create_stopwords(self, kind='lower'):
        """Create set of stopwords (lowercase or uppercase)
        lower: union of words from NLTK, sklearn, single characters
        upper: union of words from names, cities, locations
        """
        if kind == 'lower':
            stops = list(ENGLISH_STOP_WORDS)
            stops.extend(stopwords.words('english'))
            stops.extend([s for s in string.ascii_lowercase])
            # filler = _get_filler()
            # words = _get_words()

            # custom dont stopword for domain specific words 2021-01-21
            custom_allow = ['back', 'front', 'up', 'down', 'both']
            stops = [s for s in stops if s not in custom_allow]

            # custom do stopword 2021-01-21
            custom_remove = ['whilst', 'left', 'right', 'middle']
            stops.extend(custom_remove)

            wordset = frozenset(stops)  # + filler)  # + words)

        elif kind == 'upper':  # proper nouns etc
            names = self._get_names()
            # cities = get_cities1000()
            # britishlocs = get_britishlocations()
            wordset = frozenset(names)  # + cities) # + britishlocs)

        else:
            raise ValueError("kind valid in {'lower', 'upper'}")

        return wordset

    def create_bloomfilter(self, wordset):
        """Setup Bloom filter wordset"""

        bf = BloomFilter(capacity=len(wordset), error_rate=0.001)
        bf.update(wordset)

        print(
            'Filter has {:.1e} bits, capacity for {} items'.format(
                bf.num_bits, bf.capacity
            )
        )

        return bf

    def _dedupe_list_preserve_order(self, lst):
        """Extremely useful dedupe list of hashable objects, preserve order
        https://pypi.org/project/orderedset/
        """
        oset = OrderedSet(lst)
        return list(oset)

    def _join_tuples(self, lst):
        """Fairly ropey logic to rejoin tups after removing duplicates
        Needed because retained tups contain some of the words removed
        e.g. (a, {b}) [(b, c)], ({c}, d)  -> a, d
        """
        out = []
        for i, t in enumerate(lst):
            if i == 0:
                out.extend(list(t))
            elif i < (len(lst) - 1):
                if not ((t[0] == lst[i - 1][1]) & (t[1] != lst[i + 1][0])):
                    out.append(t[1])
            else:
                # if (t[0] == lt[i-1][1]):
                out.append(t[1])
        return out

    def tokenise_filter_lem(self, txt, bloom, drop_rem=True, lem=True):
        """
        Accept raw text strings, tokenise, remove all digits and punctuation,
        bloom filter and optional lemmatize
        Return list of sentences, optionally including flag for filtered words
        Maintain sentences as items in a list so downstream ngramming can occur

        NOTE: using NLTK Wordnet Lemmatizer. It's good but not ideal
              e.g. (creat)ion != (creat)e.
              # http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#WordNetCorpusReader.morphy
              TODO: try spaCy lemmatizer
        """

        # custom corrections for "-ion" words and others
        ion_to_e = set(
            [
                'laceration',
                'punctuation',
                'penetration',
                'operation',
                'irritation',
                'aggravation',
                'dislocation',
                'location',
                'amputation',
            ]
        )
        ion_to_null = set(['infection'])
        ing_to_null = set(['keyboarding', 'kerbing'])
        al_to_null = set(['corneal'])
        ly_to_null = set(['awkwardly'])
        s_to_null = set(['backwards', 'forwards'])

        def _filter_and_lem(w):
            """Filter words using bloom filter, lemmatize the rest"""
            if len(w) < 3:
                return 'REM'
            else:
                wl = w.lower()
                if wl in bloom:
                    return 'REM'
                elif lem:
                    wl_lem = self.lemtzr.lemmatize(wl, wordnet.NOUN)
                    wl_lem = self.lemtzr.lemmatize(wl_lem, wordnet.VERB)
                    wl_lem = self.lemtzr.lemmatize(wl_lem, wordnet.ADJ)
                    wl_lem = self.lemtzr.lemmatize(wl_lem, wordnet.ADV)

                    # TODO: undo wound -> wind

                    if wl_lem in ion_to_e:
                        wl_lem = wl_lem[:-3] + 'e'

                    if wl_lem in ion_to_null:
                        wl_lem = wl_lem[:-3]

                    if wl_lem in ing_to_null:
                        wl_lem = wl_lem[:-3]

                    if wl_lem in al_to_null:
                        wl_lem = wl_lem[:-1]

                    if wl_lem in ly_to_null:
                        wl_lem = wl_lem[:-2]

                    if wl_lem in s_to_null:
                        wl_lem = wl_lem[:-1]

                    return wl_lem
                else:
                    return wl

        def _remove_stutter(s):
            """Dedupe contiguous repeated words"""
            lst = list(self.ng.get_tuples_of_window(s, 2))
            lt = [t for t in lst if len(set(t)) == len(t)]
            return self._join_tuples(lt)

        def _remove_repeat_2grams(s):
            """Dedupe repeats of non/contiguous 2-grams
            e.g. apple banana carrot orange ![apple banana] grape

            NOTE: if the dupe tup exists at the end of the list,
                  it will return tup[1] at the end of the new list
                  so you get a duplicated word.
                  TODO fix this
            """
            lst = self.ng.get_tuples_of_window(s, stride=2)
            lt = self._dedupe_list_preserve_order(lst)
            return self._join_tuples(lt)

        # NOTE: using 1-gram TreeBankWord tokenisation
        lol = [word_tokenize(s) for s in sent_tokenize(txt)]
        lol = [[w for w in s if self.rx_nonchar.search(w) is None] for s in lol]
        lol = [[_filter_and_lem(w) for w in s] for s in lol]

        if drop_rem:
            lol = [[w for w in s if w != "REM"] for s in lol]

        lol = [_remove_stutter(s) for s in lol]

        # TODO: 2021-01-21 need to fix the issue where dupe is at end of list
        # lol = [_remove_repeat_2grams(s) for s in lol]

        # TODO: for the kaggle comp there are no sentences, so we can just
        # join lol to l. But elsewhere in future return lol
        return [' '.join(s) for s in lol]  # if len(s) > 0]


class NGrammer:
    def __init__(self):
        pass

    def get_tuples_of_window(self, lst, stride=2):
        """Run a window of length stride along list l
        Return generator of tuples
        https://stackoverflow.com/a/61977295/1165112
        """
        return zip(*[itertools.islice(lst, i, sys.maxsize) for i in range(stride)])

    def create_ngrams_from_lol(self, lol, max_ngram=2):
        """Create list of ngrams within list of lists (each list a sentence)
        Thus avoid creating ngrams across sentence boundaries
        Returns a single list of ngram tokens
        NOTE:
            lol should be a list of lists where each list is a sentence
            each sentence should be a list of tokens
        """
        lol_out = []
        for s in lol:
            list_of_grams = []
            for n in range(1, max_ngram + 1):
                grams = [' '.join(t) for t in self.get_tuples_of_window(s, n)]
                list_of_grams.extend(grams)
            lol_out.append(list_of_grams)

            flat_list = [x for y in lol_out for x in y]
        return flat_list
