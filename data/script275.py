
# coding: utf-8

# # Kaggle Users With The Most Forum Karma
# 
# Here, we'll calculate a Kaggle user's forum karma by summing the score (upvotes-downvotes) for each of her forum posts. We'll then show the top 100 users with the most forum karma.

# In[ ]:


from IPython.display import HTML
import pandas as pd
import re
import sqlite3

con = sqlite3.connect('../input/database.sqlite')

users = pd.read_sql_query("""
SELECT u.Id UserId,
       u.DisplayName,
       SUM(m.Score) ForumKarma,
       COUNT(DISTINCT m.Id) NumForumPosts,
       1.0*SUM(m.Score)/COUNT(DISTINCT m.Id) KarmaPerPost
FROM ForumMessages m
INNER JOIN Users u ON m.AuthorUserId=u.Id
GROUP BY u.Id
ORDER BY SUM(m.Score) DESC
LIMIT 100""", con)

users["User"] = ""

for i in range(len(users)):
    users.loc[i, "User"] = "<" + "a href='https://www.kaggle.com/u/" + str(users["UserId"][i]) + "'>" + users["DisplayName"][i] + "<" + "/a>"

users.index = range(1, len(users)+1)
pd.set_option("display.max_colwidth", -1)

HTML(users[["User", "ForumKarma", "NumForumPosts", "KarmaPerPost"]].to_html(escape=False))

