from sklearn.feature_extraction.text import CountVectorizer

text = ["London Paris London", "Paris Paris London"]

# find the count between the text item

cv = CountVectorizer()

count_matrix = cv.fit_transform(text) #function in the sklearn to count
print(count_matrix.toarray()) #gives us the points for the cosine similiaty 
#output [[2 1][1 2]]


