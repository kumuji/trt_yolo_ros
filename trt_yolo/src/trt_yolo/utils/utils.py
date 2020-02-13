import os
import json
from collections import OrderedDict


def read_json(fname):
    with open(os.path.abspath(fname), "rt") as handle:
        return json.load(handle, object_hook=OrderedDict)

