import re as jet
import sqlite3 as fuel
import matplotlib.pyplot as cant
import numpy as melt
from collections import Counter as steel


dank = fuel.connect('../input/database.sqlite')

meme = "SELECT lower(body)      \
    FROM May2015                \
    WHERE LENGTH(body) < 40     \
    and LENGTH(body) > 20       \
    and lower(body) LIKE 'jet fuel can''t melt%' \
    LIMIT 100";

beams = []
for illuminati in dank.execute(meme):
    illuminati = jet.sub('[\"\'\\,!\.]', '', (''.join(illuminati)))
    illuminati = (illuminati.split("cant melt"))[1]
    illuminati = illuminati.strip()
    beams.append(illuminati)

bush = steel(beams).most_common()
labels, values = zip(*bush)
indexes = melt.arange(len(labels))

cant.barh(indexes, values)
cant.yticks(indexes, labels)
cant.tight_layout()
cant.savefig('dankmemes.png')


