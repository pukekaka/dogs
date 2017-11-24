from gensim.models import word2vec

filename = 'E:/Works/Data/samples/output_c/basicblock_by_space'

fn = word2vec.Text8Corpus(filename)
model = word2vec.Word2Vec(fn, size=100, window=5, min_count=5, workers=4)

result = model.most_similar(positive=['pusheax', 'movedieax'], negative=['movecx4'], topn=5)
for el in result:
    print (el[0])