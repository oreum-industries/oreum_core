# Copyright 2026 Oreum Industries
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

__all__ = ["SnakeyLowercaser"]


class SnakeyLowercaser:
    """Clean and standardise a string to snakey_lowercase
    Split CamelCasedStrings
    Convert '[-/.]' to '_' and preserve existing '_',
    Useful for the often messy column names present in Excel tables
    """

    def __init__(self, force_to_underscore: str = None, allowed_punct: str = None):
        """Init and setup lots of regexes
        force_to_underscore: include these strings in the force to underscore.
            By default already includes "/." because we never want these to make
            it through strong cleaning. You might want to include force hyphens,
            em dashes etc etc to underscores too
        allowed_punct: include these punctuation strings.
            By default already includes "_" because we want to preserve
            underscores (although only 1 in a row)
        """
        force_to_underscore = "".join(filter(None, ["/.", force_to_underscore]))
        self.rx_to_underscore = re.compile(
            "[{}]".format(re.escape(force_to_underscore))
        )

        allowed_punct = "".join(filter(None, ["_", allowed_punct]))
        remove_punct = re.sub(
            "{}".format(re.escape(allowed_punct)), "", string.punctuation
        )
        self.rx_punct = re.compile("[{}]".format(re.escape(remove_punct)))

        self.rx_splitter1 = re.compile(r"([A-Za-z0-9])([A-Z][a-z]+)")
        self.rx_patsy_factor = re.compile(r"^(.*)(\[T\.|\[)(.*)(\])(.*)$")
        self.rx_patsy_numpy = re.compile(r"^np\.(.*)\((.*)\)$")
        self.rx_patsy_interaction = re.compile(r":")
        self.rx_multi_underscore = re.compile(r"[_]{2,}")
        self.rx_underscore_startend = re.compile(r"^\_|\_$")

    def clean(self, s: str) -> str:
        """Clean strings essential"""
        s0 = self.rx_to_underscore.sub(r"_", str(s).strip().lower())
        s1 = self.rx_punct.sub("", s0)
        s2 = self.rx_splitter1.sub(r"\1_\2 ", s1)
        s3 = "_".join(s2.split())
        s4 = self.rx_multi_underscore.sub(r"_", s3)
        s5 = self.rx_underscore_startend.sub(r"", s4)
        return s5

    def clean_patsy(self, s: str) -> str:
        """Clean strings for patsy features"""
        s0 = str(s).replace("-", "_")
        if len(f := self.rx_patsy_factor.findall(s0)) > 0:
            s0 = f[0][0] + "_t_" + f[0][2] + f[0][4]
        # run twice as a very lazy alt to recursion to cover interaction of 2 cats
        if len(f := self.rx_patsy_factor.findall(s0)) > 0:
            s0 = f[0][0] + "_t_" + f[0][2] + f[0][4]
        if len(f := self.rx_patsy_numpy.findall(s0)) > 0:
            s0 = f[0][1] + "_" + f[0][0]
        s1 = self.rx_patsy_interaction.sub(r"_x_", s0)
        s2 = self.rx_punct.sub(r"", s1)
        s3 = self.rx_splitter1.sub(r"\1_\2 ", s2)
        s4 = "_".join(s3.lower().split())
        s5 = self.rx_multi_underscore.sub(r"_", s4)
        return s5
