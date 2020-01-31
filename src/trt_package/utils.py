import os
import json
from collections import OrderedDict


def read_json(fname):
    with os.path(fname).open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)
