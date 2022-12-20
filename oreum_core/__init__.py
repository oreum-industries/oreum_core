# copyright 2022 Oreum Industries
import logging

__version__ = "0.4.8"

# logger goes to null handler by default. importing packages can set elsewhere
logging.getLogger('oreum_core').addHandler(logging.NullHandler())
