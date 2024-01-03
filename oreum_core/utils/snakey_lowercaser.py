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

# utils.snakey_lowercaser.py
"""SnakeyLowercaser used everywhere"""
import re
import string

__all__ = ['SnakeyLowercaser']


class SnakeyLowercaser:
    """Clean and standardise a string to snakey_lowercase
    Split CamelCasedStrings
    Convert '[-/.]' to '_' and preserve existing '_',
    Useful for the often messy column names present in Excel tables
    """

    def __init__(self):
        """Init and setup lots of regexes"""
        punct_to_remove = re.sub(r'[_]', '', string.punctuation)
        self.rx_to_underscore = re.compile(r'[-/.]')
        self.rx_punct = re.compile('[{}]'.format(re.escape(punct_to_remove)))
        self.rx_splitter1 = re.compile(r'([A-Za-z0-9])([A-Z][a-z]+)')
        self.rx_patsy_factor = re.compile(r'^(.*)(\[T\.|\[)(.*)(\])(.*)$')
        self.rx_patsy_numpy = re.compile(r'^np\.(.*)\((.*)\)$')
        self.rx_patsy_interaction = re.compile(r':')
        self.rx_multi_underscore = re.compile(r'[_]{2,}')

    def clean(self, s: str) -> str:
        """Clean strings essential"""
        s0 = self.rx_to_underscore.sub('_', str(s))
        s1 = self.rx_punct.sub('', s0)
        s2 = self.rx_splitter1.sub(r'\1_\2 ', s1)
        s3 = '_'.join(s2.lower().split())
        s4 = self.rx_multi_underscore.sub('_', s3)
        return s4

    def clean_patsy(self, s: str) -> str:
        """Clean strings for patsy features"""
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
