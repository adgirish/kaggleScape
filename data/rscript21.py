import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# This script identifies which communication styles receive highest ranks
# For illustration purposes I defined 3 styles such as Passive, Assertive and Aggressive
# The list of key words must of course be extended

sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql("SELECT score, body FROM May2015 WHERE LENGTH(body) > 5 AND LENGTH(body) < 100 LIMIT 10000", sql_conn)
    
keywords = pd.DataFrame({'Passive': pd.Series(['if you have the time','hmm','well','that was my fault','not sure']),
                         'Assertive': pd.Series(['good idea','great idea','thanks for','good to know','really like', 'too','sorry for']),
                         'Aggressive': pd.Series(['I shot','fuck','fucking','ass','idiot'])})

content_summary = pd.DataFrame()
for col in keywords:
    content = df[df.body.apply(lambda x: any(keyword in x.split() for keyword in keywords[col]))]
    content_summary[col] = content.describe().score

keys = content_summary.keys()

content_summary = content_summary.transpose()

# Setting the positions and width for the bars
pos = list(range(len(content_summary['count'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

clrs = []
for v in content_summary['mean'].values:
    if v < 2:
        clrs.append('#FFC1C1')
    elif v < 5:
        clrs.append('#F08080')
    elif v < 10:
        clrs.append('#EE6363')
    else:
        clrs.append('r')

plt.bar(pos,
        content_summary['count'],
        width,
        alpha=0.5,
        # with color
        color=clrs,
        label=keys)

# Set the y axis label
ax.set_ylabel('Number of comments')

# Set the chart's title
ax.set_title('Which communication style receives highest ranks?')

# Set the position of the x ticks
ax.set_xticks([p + 0.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(keys)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(content_summary['count'])+20])

rects = ax.patches

# Now make some labels
for ii,rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'% ("Score {0:.2f}".format(content_summary['mean'][ii])),
                 ha='center', va='bottom')

plt.grid()

plt.savefig("CommunicationStyles.png")