from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import euclidean_distances
import sys, codecs
from pyemd import emd
import sklearn.metrics
import nltk
import re

n_topics = 500
NOUN = ['NN', 'NNS', 'NNP', 'NNPS']
VERB = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ADJ = ['JJ', 'JJR', 'JJS']
ADV = ['RB', 'RBR', 'RBS']

vector_file = "/home/tong/Documents/python/glove.42B.300d.txt"



def score_tfidf_cosine(src, dst):
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


def score_tfidf_euclidean(src, dst):
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

	res = [round(spatial.distance.euclidean(b1_vecs[i], b2_vecs[i]),2) for i in range(lines)]
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))


def score_lsa(src, dst):
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
	vectorizer = TfidfVectorizer()
	vectors=vectorizer.fit_transform(b1 + b2)
	len(vectorizer.vocabulary_)
	svd = TruncatedSVD(n_topics)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)
	X = lsa.fit_transform(vectors)	
	print X.shape
	b1_v = vectorizer.transform(b1)
	b2_v = vectorizer.transform(b2)
	b1_vecs = lsa.transform(b1_v)
	b2_vecs = lsa.transform(b2_v)

	res = [round(5*(1 - spatial.distance.cosine(b1_vecs[i], b2_vecs[i])),2) for i in range(lines)]
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))

def score_bow(src, dst):
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

	vectorizer = CountVectorizer()
	vectors=vectorizer.fit_transform(b1 + b2)
	b1_vecs = vectorizer.transform(b1).todense()
	b2_vecs = vectorizer.transform(b2).todense()

	res = [round(5*(1 - spatial.distance.cosine(b1_vecs[i], b2_vecs[i])),2) for i in range(lines)]
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))


def score_lda(src, dst):
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

	vectorizer = CountVectorizer()
	vectors=vectorizer.fit_transform(b1 + b2)

	lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
	X = lda.fit_transform(vectors)
	print X.shape
	b1_v = vectorizer.transform(b1)
	b2_v = vectorizer.transform(b2)
	b1_vecs = lda.transform(b1_v)
	b2_vecs = lda.transform(b2_v)

	res = [round(5*(1 - spatial.distance.cosine(b1_vecs[i], b2_vecs[i])),2) for i in range(lines)]
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))


def score_word2vec_tfidf(src, dst, wv):
	b1 = []
	b2 = []
	lines = 0
	with open(src) as p:
		for i, line in enumerate(p):
			s = line.split('\t')
			b1.append(s[0])
			b2.append(s[1][:-1]) #remove \n
			lines = i + 1

	vectorizer = TfidfVectorizer()
	vectors=vectorizer.fit_transform(b1 + b2)
	vs = np.zeros((vectors.shape[1],300))
	for word in vectorizer.vocabulary_.keys():
		if word in wv:
			vs[vectorizer.vocabulary_[word]] = wv[word]

	v = np.zeros((vectors.shape[0], 300))
	b1_v = vectorizer.transform(b1)
	b2_v = vectorizer.transform(b2)
	b1_vecs = np.dot(b1_v.todense(), vs)
	b2_vecs = np.dot(b2_v.todense(), vs)

	res = [round(5*(1 - spatial.distance.cosine(b1_vecs[i], b2_vecs[i])),2) for i in range(lines)]
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))

def readVectors(vector_file):
	n_words = 100000

	numpy_arrays = []
	labels_array = []
	g = {}
	with codecs.open(vector_file, 'r', 'utf-8') as f:
		for c, r in enumerate(f):
			sr = r.split()
			labels_array.append(sr[0])
			vec = np.array([float(i) for i in sr[1:]])
			numpy_arrays.append(vec)
			g[sr[0]] = vec
			if c == n_words:
				break
	return numpy_arrays, labels_array, g

def score_glove_tfidf(src, dst, numpy_arrays, labels_array):
	b1 = []
	b2 = []
	lines = 0
	with open(src) as p:
		for i, line in enumerate(p):
			s = line.split('\t')
			b1.append(s[0])
			b2.append(s[1][:-1]) #remove \n
			lines = i + 1

	vectorizer = TfidfVectorizer()
	vectors=vectorizer.fit_transform(b1 + b2)

	vs = np.zeros((vectors.shape[1],300))
	for word in vectorizer.vocabulary_.keys():
		if word in labels_array:
			vs[vectorizer.vocabulary_[word]] = numpy_arrays[labels_array.index(word)]

	v = np.zeros((vectors.shape[0], 300))
	b1_v = vectorizer.transform(b1)
	b2_v = vectorizer.transform(b2)
	b1_vecs = np.dot(b1_v.todense(), vs)
	b2_vecs = np.dot(b2_v.todense(), vs)

	res = [round(5*(1 - spatial.distance.cosine(b1_vecs[i], b2_vecs[i])),2) for i in range(lines)]
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))

def score_word2vec_wmd(src, dst, wv):
	b1 = []
	b2 = []
	lines = 0
	with open(src) as p:
		for i, line in enumerate(p):
			s = line.split('\t')
			b1.append(s[0])
			b2.append(s[1][:-1]) #remove \n
			lines = i + 1

	vectorizer = CountVectorizer()
	vectors=vectorizer.fit_transform(b1 + b2)
	common = [word for word in vectorizer.get_feature_names() if word in wv]
	W_common = [wv[w] for w in common]
	vectorizer = CountVectorizer(vocabulary=common, dtype=np.double)
	b1_v = vectorizer.transform(b1)
	b2_v = vectorizer.transform(b2)

	D_ = sklearn.metrics.euclidean_distances(W_common)
	D_ = D_.astype(np.double)
	D_ /= D_.max()

	b1_vecs = b1_v.toarray()
	b2_vecs = b1_v.toarray()
	b1_vecs /= b1_v.sum()
	b2_vecs /= b2_v.sum()
	b1_vecs = b1_vecs.astype(np.double)
	b2_vecs = b2_vecs.astype(np.double)

	res = [round(emd(b1_vecs[i], b2_vecs[i], D_),2) for i in range(lines)]
	
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))
	print src + ' finished!'


def normarlize_score(res):
	s = res[:]
	s.sort()
	l = len(s)
	for i in xrange(l):
		if res[i] >= s[0] and res[i] < s[l/6]:
			res[i] = 0
		elif res[i] >= s[l/6] and res[i] < s[l/3]:
			res[i] = 1
		elif res[i] >= s[l/3] and res[i] < s[l/2]:
			res[i] = 2
		elif res[i] >= s[l/2] and res[i] < s[2*l/3]:
			res[i] = 3
		elif res[i] >= s[2*l/3] and res[i] < s[5*l/6]:
			res[i] = 4
		else:
			res[i] = 5
	return res

def score_word2vec_pos(src, dst, wv, normalize=True):
	b1 = []
	b2 = []
	lines = 0
	with open(src) as p:
		for i, line in enumerate(p):
			s = line.split('\t')
			b1.append(s[0])
			b2.append(s[1][:-1]) #remove \n
			lines = i + 1

	b1_pos = [nltk.pos_tag(nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+',' ', text))) for text in b1]
	b2_pos = [nltk.pos_tag(nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+',' ', text))) for text in b2]

	res = []
	for i in range(lines):
		tags1_noun = [tag[0] for tag in b1_pos[i] if tag[1] in NOUN]
		tags2_noun = [tag[0] for tag in b2_pos[i] if tag[1] in NOUN]
		tags1_verb = [tag[0] for tag in b1_pos[i] if tag[1] in VERB]
		tags2_verb = [tag[0] for tag in b2_pos[i] if tag[1] in VERB]
		tags1_adj = [tag[0] for tag in b1_pos[i] if tag[1] in ADJ]
		tags2_adj = [tag[0] for tag in b2_pos[i] if tag[1] in ADJ]
		tags1_adv = [tag[0] for tag in b1_pos[i] if tag[1] in ADV]
		tags2_adv = [tag[0] for tag in b2_pos[i] if tag[1] in ADV]
		r_noun = [wv.similarity(tag1, tag2) for tag1 in tags1_noun for tag2 in tags2_noun if tag1 in wv and tag2 in wv]
		r_verb = [wv.similarity(tag1, tag2) for tag1 in tags1_verb for tag2 in tags2_verb if tag1 in wv and tag2 in wv]
		r_adj = [wv.similarity(tag1, tag2) for tag1 in tags1_adj for tag2 in tags2_adj if tag1 in wv and tag2 in wv]
		r_adv = [wv.similarity(tag1, tag2) for tag1 in tags1_adv for tag2 in tags2_adv if tag1 in wv and tag2 in wv]
		r = []
		if len(r_noun) != 0
			r.append(max(r_noun))
		if len(r_verb) != 0:
			r.append(max(r_verb))
		if len(r_adj) != 0:
			r.append(max(r_adj))
		if len(r_adv) != 0:
			r.append(max(r_adv))
		if len(r) == 0:
			res.append(-1)
		else:
			res.append(round(5*sum(r)/len(r), 2))

	if normalize:
		res = normarlize_score(res)

	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))
	print src + ' finished!'

def score_glove_pos(src, dst, numpy_arrays, labels_array, g, normalize=True):
	b1 = []
	b2 = []
	lines = 0
	with open(src) as p:
		for i, line in enumerate(p):
			s = line.split('\t')
			b1.append(s[0])
			b2.append(s[1][:-1]) #remove \n
			lines = i + 1

	b1_pos = [nltk.pos_tag(nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+',' ', text))) for text in b1]
	b2_pos = [nltk.pos_tag(nltk.word_tokenize(re.sub(r'[^\x00-\x7F]+',' ', text))) for text in b2]

	res = []
	for i in range(lines):
		tags1 = [tag[0] for tag in b1_pos[i] if tag[1] in NOUN]
		tags2 = [tag[0] for tag in b2_pos[i] if tag[1] in NOUN]
		r = [1 - spatial.distance.cosine(g[tag1], g[tag2]) for tag1 in tags1 for tag2 in tags2 if tag1 in labels_array and tag2 in labels_array]
		if len(r) == 0:
			res.append(0)
		else:
			res.append(round(5*max(r), 2))

	if normalize:
		res = normarlize_score(res)
			
	with open(dst, 'w') as thefile:
		thefile.write("\n".join(str(i) for i in res))
	print src + ' finished!'






# score_tfidf_cosine('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum')
# score_tfidf_cosine('./dataset/STS.input.answers-students.txt', './dataset/sys.students')
# score_tfidf_cosine('./dataset/STS.input.belief.txt', './dataset/sys.belief')
# score_tfidf_cosine('./dataset/STS.input.headlines.txt', './dataset/sys.headlines')
# score_tfidf_cosine('./dataset/STS.input.images.txt', './dataset/sys.images')
# score_tfidf_euclidean('./dataset/STS.input.belief.txt', './dataset/sys.belief')
# score_tfidf_euclidean('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum')
# score_tfidf_euclidean('./dataset/STS.input.answers-students.txt', './dataset/sys.students')
# score_tfidf_euclidean('./dataset/STS.input.headlines.txt', './dataset/sys.headlines')
# score_tfidf_euclidean('./dataset/STS.input.images.txt', './dataset/sys.images')
# score_lsa('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum')
# score_lsa('./dataset/STS.input.answers-students.txt', './dataset/sys.students')
# score_lsa('./dataset/STS.input.belief.txt', './dataset/sys.belief')
# score_lsa('./dataset/STS.input.headlines.txt', './dataset/sys.headlines')
# score_lsa('./dataset/STS.input.images.txt', './dataset/sys.images')

# score_bow('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum')
# score_bow('./dataset/STS.input.answers-students.txt', './dataset/sys.students')
# score_bow('./dataset/STS.input.belief.txt', './dataset/sys.belief')
# score_bow('./dataset/STS.input.headlines.txt', './dataset/sys.headlines')
# score_bow('./dataset/STS.input.images.txt', './dataset/sys.images')

# score_lda('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum')
# score_lda('./dataset/STS.input.answers-students.txt', './dataset/sys.students')
# score_lda('./dataset/STS.input.belief.txt', './dataset/sys.belief')
# score_lda('./dataset/STS.input.headlines.txt', './dataset/sys.headlines')
# score_lda('./dataset/STS.input.images.txt', './dataset/sys.images')

# wv = Word2Vec.load_word2vec_format("/home/tong/Documents/python/GoogleNews-vectors-negative300.bin.gz", binary = True)
# score_word2vec_tfidf('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum', wv)
# score_word2vec_tfidf('./dataset/STS.input.answers-students.txt', './dataset/sys.students', wv)
# score_word2vec_tfidf('./dataset/STS.input.belief.txt', './dataset/sys.belief', wv)
# score_word2vec_tfidf('./dataset/STS.input.headlines.txt', './dataset/sys.headlines', wv)
# score_word2vec_tfidf('./dataset/STS.input.images.txt', './dataset/sys.images', wv)



# numpy_arrays, labels_array, g = readVectors(vector_file)
# score_glove_tfidf('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum', numpy_arrays, labels_array)
# score_glove_tfidf('./dataset/STS.input.answers-students.txt', './dataset/sys.students', numpy_arrays, labels_array)
# score_glove_tfidf('./dataset/STS.input.belief.txt', './dataset/sys.belief', numpy_arrays, labels_array)
# score_glove_tfidf('./dataset/STS.input.headlines.txt', './dataset/sys.headlines', numpy_arrays, labels_array)
# score_glove_tfidf('./dataset/STS.input.images.txt', './dataset/sys.images', numpy_arrays, labels_array)

# wv = Word2Vec.load_word2vec_format("/home/tong/Documents/python/GoogleNews-vectors-negative300.bin.gz", binary = True)
# print "done" + " loading"
# score_word2vec_wmd('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum', wv)
# score_word2vec_wmd('./dataset/STS.input.answers-students.txt', './dataset/sys.students', wv)
# score_word2vec_wmd('./dataset/STS.input.belief.txt', './dataset/sys.belief', wv)
# score_word2vec_wmd('./dataset/STS.input.headlines.txt', './dataset/sys.headlines', wv)
# score_word2vec_wmd('./dataset/STS.input.images.txt', './dataset/sys.images', wv)


wv = Word2Vec.load_word2vec_format("/home/tong/Documents/python/GoogleNews-vectors-negative300.bin.gz", binary = True)
print "done" + " loading"
normalize = True
score_word2vec_pos('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum', wv, normalize)
score_word2vec_pos('./dataset/STS.input.answers-students.txt', './dataset/sys.students', wv, normalize)
score_word2vec_pos('./dataset/STS.input.belief.txt', './dataset/sys.belief', wv, normalize)
score_word2vec_pos('./dataset/STS.input.headlines.txt', './dataset/sys.headlines', wv, normalize)
score_word2vec_pos('./dataset/STS.input.images.txt', './dataset/sys.images', wv, normalize)


# numpy_arrays, labels_array, g = readVectors(vector_file)
# print "done" + " loading"
# score_glove_pos('./dataset/STS.input.answers-forums.txt', './dataset/sys.forum', numpy_arrays, labels_array, g, normalize=False)
# score_glove_pos('./dataset/STS.input.answers-students.txt', './dataset/sys.students', numpy_arrays, labels_array, g, normalize=False)
# score_glove_pos('./dataset/STS.input.belief.txt', './dataset/sys.belief', numpy_arrays, labels_array, g, normalize=False)
# score_glove_pos('./dataset/STS.input.headlines.txt', './dataset/sys.headlines', numpy_arrays, labels_array, g, normalize=False)
# score_glove_pos('./dataset/STS.input.images.txt', './dataset/sys.images', numpy_arrays, labels_array, g, normalize=False)