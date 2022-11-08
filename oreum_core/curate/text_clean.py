# curate.text_clean.py
# copyright 2022 Oreum Industries
import string

import ftfy
import numpy as np
import regex as re

__all__ = ['SnakeyLowercaser', 'TextCleaner']


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

    def clean(self, s: str) -> str:
        s0 = self.rx_to_underscore.sub('_', str(s))
        s1 = self.rx_punct.sub('', s0)
        s2 = self.rx_splitter1.sub(r'\1_\2 ', s1)
        s3 = '_'.join(s2.lower().split())
        s4 = self.rx_multi_underscore.sub('_', s3)
        return s4

    def clean_patsy(self, s: str) -> str:
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

    def fix_unicode(self, txt: str) -> str:
        """Fix bad unicode / emojis etc and try to remove crud"""

        t = ftfy.fix_text(txt, fix_character_width=False)  # fix encoding
        t = self.rx_hex.sub('', t)  # remove hex like '=b7', '=f5' etc

        return t

    def basic_clean(self, txt: str) -> str:
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

    def convert_bad_number_representation_to_float(self, s: str) -> float:
        """Accept a string that represents a number (poorly), convert to float
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
        s0 = self.rx_number_junk.sub('', str(s).strip().lower())
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
