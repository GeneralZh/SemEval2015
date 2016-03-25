from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

def score_tfidf(src, dst):
	##read sentence pairs to two lists
	b1 = []
	b2 = []
	lines = 0
	with open(src) as p:
		for i, line in enumerate(p):
			s = line.split('\t')
			b1.append(s[0])
			b2.append(s[1][:-1]) #remove \n
			lines = i + 1

	# vectorizer = TfidfVectorizer(stop_words = 'english') #if remove stop words, some sentence becomes 0
	vectorizer = TfidfVectorizer()
	vectorizer.fit_transform(b1 + b2)
	b1_vecs = vectorizer.transform(b1).todense()
	b2_vecs = vectorizer.transform(b2).todense()

	res = [round(5*(1 - spatial.distance.cosine(b1_vecs[i], b2_vecs[i])),2) for i in range(lines)]

	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))


score_tfidf('./dataset/STS.input.belief.txt', './dataset/sys.belief')
score_tfidf('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum')
score_tfidf('./dataset/STS.input.answers-students.txt', './dataset/sys.students')
score_tfidf('./dataset/STS.input.headlines.txt', './dataset/sys.headlines')
score_tfidf('./dataset/STS.input.images.txt', './dataset/sys.images')