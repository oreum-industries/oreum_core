**Oreum Industries**

# Oreum Core Tools `oreum_core`

This is an ever-growing package of core tools for use on client projects.

NOTES:

+ Uses a scientific Python stack for scripting
+ Hosted on
[Oreum Industries' GitHub](https://github.com/oreum-industries/oreum_core)
+ Project began on 2021-01-01
+ The README.md is MacOS and POSIX oriented
+ See LICENCE.md for licensing and copyright details
+ See CONTRIBUTORS.md for list of contributors

FURTHER NOTES:

+ This package is a work in progress (v0.y.z) and liable to breaking changes,
and inconveniences to the user.
+ This package is solely designed for ease of use and rapdi development by
employees of Oreum Industries, and selected clients. It is not intended for
public usage and will not be supported in that fashion.

POST INSTALL:

Currently requires post-install manual step:

1. Download and install NLTK files, e.g.

```python
import nltk
nltk.downloader.download(['stopwords', 'treebank', 'wordnet','punkt'])
```

---

Copyright 2021 Oreum OÜ t/a Oreum Industries. All rights reserved.
See LICENSE.md.

Oreum OÜ t/a Oreum Industries, Sepapaja 6, Tallinn, 15551, Estonia,
reg.16122291, [oreum.io](https://oreum.io)

---
Oreum OÜ &copy; 2021
