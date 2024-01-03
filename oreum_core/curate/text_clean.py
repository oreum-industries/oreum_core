# Copyright 2024 Oreum Industries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# curate.text_clean.py
"""Text Cleaning"""
import re
import string

import ftfy
import numpy as np

__all__ = ['TextCleaner']


class TextCleaner:
    """Potentially useful text cleaing - strips lots of crud.
    Originally developed a number of years ago, so may need revision
    """

    def __init__(self):
        """Init with lots of regexes"""
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
