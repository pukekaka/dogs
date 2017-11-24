from gensim.models import word2vec

import time
import zipfile
import os

start = time.time()

current_directory = os.path.dirname(os.path.abspath(__file__))
# directory = 'zipdata'
# filename = 'APT_basicblock_by_space_replace.zip'
# filepath = os.path.join(current_directory, directory ,filename)
output_path = os.path.join(current_directory, 'model', 'basicblock_by_space_replace')

filename = 'E:/Works/Data/samples/output_c/basicblock_by_space_replace'

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = f.read(f.namelist()[0])

  return data

# data = read_data(filepath)


# t8c = word2vec.Text8Corpus(data)
ls = word2vec.LineSentence(filename)

# model = word2vec.Word2Vec(t8c, size=100, window=5, min_count=5, workers=4)
model = word2vec.Word2Vec(ls, size=200, window=10, hs=1, min_count=2, sg=1)

model.save(output_path)
# model.save("./model/basicblock_by_space.model")

result = model.most_similar(positive=['pusheax'], topn=5)
for el in result:
    print (el)

# result = model.most_similar(positive=['pushedi'], topn=20)
# for el in result:
#     print (el)

end = time.time() - start

print('time', end)

