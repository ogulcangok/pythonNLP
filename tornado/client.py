import tornado
import tornado.ioloop
import tornado.web
import os, uuid
import pandas as pd
import re
import numpy as np

__UPLOADS__ = "uploads/"

patterns = ['(.*)\s\(','(.*)\s-','(.*)\.co','(.*)\s-']
shingles = []
duplicates = []

def get_shingles(text, char_ngram=5):
    """Create a set of overlapping character n-grams.
    
    Only full length character n-grams are created, that is the first character
    n-gram is the first `char_ngram` characters from text, no padding is applied.

    Each n-gram is spaced exactly one character apart.

    Parameters
    ----------

    text: str
        The string from which the character n-grams are created.

    char_ngram: int (default 5)
        Length of each character n-gram.
    """
    return set(text[head:head + char_ngram] for head in range(0, len(text) - char_ngram))
def jaccard(set_a, set_b):
    """Jaccard similarity of two sets.
    
    The Jaccard similarity is defined as the size of the intersection divided by
    the size of the union of the two sets.

    Parameters
    ---------
    set_a: set
        Set of arbitrary objects.

    set_b: set
        Set of arbitrary objects.
    """
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)

def extract_source(source):
    """
    splits the './' etc. from the source column
    alsı the on Sunday and 'The'
    """
    res = re.split('\(|\.|\-',source)[0] 
    res =res.lower()
    res = res.replace('on sunday','') 
    res = res.replace('sunday','') 
    res = re.sub('the\s','',res) 
    res = re.sub('daily\s','',res) 
    out = res.strip() if res else source.strip()
    return(out)
    
    
our_special_word = 'nonascii'
def remove_ascii_words(dat):
  non_ascii_words = []
  for i in range(len(dat)):
        for word in dat.loc [i, 'content'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                dat.loc [i, 'content'] = dat.loc[i, 'content'].replace(word, our_special_word)
  return non_ascii_words

class Userform(tornado.web.RequestHandler):
    def get(self):
        self.render("fileuploadform.html")


class Upload(tornado.web.RequestHandler):
    def post(self):
        fileinfo = self.request.files['filearg'][0]
        print ("fileinfo is", fileinfo)
        fname = fileinfo['filename']
        extn = os.path.splitext(fname)[1]
        fname = fname.replace('.json','')
        cname = fname + extn
       
        dat = pd.read_json("/home/ogulcan/tornado/uploads/UK_afterJaccard.json")
        for news in dat['content']: shingles.append(get_shingles(news.lower()))

        for i_doc in range(len(shingles)):
  
            for j_doc in range(i_doc + 1, len(shingles)):
                jaccard_similarity = jaccard(shingles[i_doc], shingles[j_doc])
                is_duplicate = jaccard_similarity >= 0.75
                if is_duplicate:
            
                    duplicates.append((i_doc, j_doc, jaccard_similarity))
         
         
        jac = pd.DataFrame(duplicates,columns=("Doc 1","Doc2","Simil"))
        dat = dat.drop(dat.index[jac['Doc2']])
        dat['source'] = dat['source'].apply(lambda x: extract_source(x)) 

        document_lengths = np.array(list(map(len, dat.content.str.split(' '))))
        dat[document_lengths <= 50]

        dat = dat[document_lengths > 50] 
        data = dat.content.values.tolist()
        data = [re.sub('\S*@\S*\s?', '', content) for content in data]
        data = [re.sub('\s+', ' ', content) for content in data]
        data = [re.sub('\n+', ' ', content) for content in data]
        data = [re.sub('\r+', ' ', content) for content in data]   
        data = [re.sub("\'", "", content) for content in data]
        fileName = fname + "_Cleaned.json"
        dat.to_json("uploads/"+fileName)
        self.finish(fileName + " is uploaded!! Check %s folder" %__UPLOADS__)
        



application = tornado.web.Application([
        (r"/", Userform),
        (r"/upload", Upload),
        ], debug=True)

    
    


if __name__ == "__main__":
    application.listen(8000)
    tornado.ioloop.IOLoop.instance().start()
    
    
    
