import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

EXCLUDED_BIGRAMS = [
"et al",
"10 10",
"international conference",
"neural information",
"information processing",
"processing systems",
"advances neural",
"supplementary material"
]

# You can read a CSV file like this
papers = pd.read_csv("../input/Papers.csv")

cv = CountVectorizer(ngram_range=(2,2), max_features = 500, stop_words='english')
cv.fit(papers.PaperText)

X = cv.transform(papers.PaperText)
counts = X.sum(axis=0)

df = pd.DataFrame({'Bigrams': cv.get_feature_names(), 'Count': counts.tolist()[0]})
df = df[df.Bigrams.map(lambda x: x not in EXCLUDED_BIGRAMS)]
df.sort_values(by='Count', ascending=False, inplace=True)

print(df.head(50))