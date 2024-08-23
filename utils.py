import numpy as np
import math
import ujson as json

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2, binom
from collections import Counter
from nltk.util import ngrams 
from itertools import product
from collections import defaultdict



### DATA LOADING & PREPROCESSING ###
#bert_tok = AutoTokenizer.from_pretrained("bert-base-cased")
#bert_word = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
#model_w2v= api.load("word2vec-google-news-300")
#model_glove = api.load("glove-wiki-gigaword-50")
#model_sent = SentenceTransformer('bert-base-nli-mean-tokens')
#model_sent = SentenceTransformer('paraphrase-albert-small-v2')
#model_sent = SentenceTransformer('paraphrase-MiniLM-L3-v2')
#metric = load_metric("bertscore")
#emb4bert = Embedding4BERT("bert-base-uncased")
#modelcls=sentence_similarity(model_name='bert-base-uncased',embedding_type='cls_token_embedding')


class Sentence():
    """
    Object with information about a single sentence.

    ...

    Attributes
    ----------
    sent_object :  stanza.models.common.doc.Sentence
        Stanza sentence object with parse information
    sentence_string: str
        Sentence string.
    tokens: list
        Tokens collected with Stanza parser.

    Methods
    -------
    _clear_dots(input_string)
        Clear noise from Webis-type sentences.
    _clear_caps(input_string)
        Remove capital letter substring (usually noise) from the input string.
    is_real_sentence()
        Change the photo's gamma exposure.

    """
    def __init__(self,sent_object,sentence_string):
        #self.sentence = self._clear_caps(sentence_string)
        self.sentence = self._clear_dots(sentence_string)
        self.sent_object = sent_object
        self.tokens = [x.lemma.lower() for x in sent_object.words]
    
    def _clear_dots(self,input_string):
        """
        Clear noise from Webis-type sentences.

        Parameters
        ----------
        input_string : str
            string to clean.

        Returns
        -------
        str
            cleaned string.
        """
        return input_string.replace("..."," ")
    
    def _clear_caps(self,input_string):
        """
        Remove capital letter substring (usually noise) from the input string.

        Parameters
        ----------
        input_string : str
            string to clean.

        Returns
        -------
        str
            cleaned string.
        """
        return ''.join(ch for ch in input_string if not ch.isupper())

    def is_real_sentence(self):
        """
        Checks if the sentence has a predicate.

        Returns
        -------
        bool
            True if sentence contains a predicate, otherwise False.
        """
        upos_tags = [x.upos for x in self.sent_object.words]
        if "VERB" in upos_tags or "AUX" in upos_tags:
            return True
        else:
            return False

class Sample():
    """
    Object with information about a single sample in the dataset.

    ...

    Attributes
    ----------
    json_sample :  dict
        json-style dictionary containing input sample with two paragraphs;
        has form {"paragraph_1":"","paragraph_2":""}
    par1: str
        First paraphraph.
    par2: str
        Second paragraph.
    to_align: bool
        Whether sample meets criteria to be aligned.
    one2many: bool
        Whether sample contains one2many mappings only 
    alignment: list
        List of output alignments.
    output_sentences: list
        #


    Methods
    -------
    golden_answer()
        Returns golden answer for the sample (if it is provided in the dataset)
    preprocess()
        Preprocess sample to check if it should be aligned and output sentences for each paragraph.
    _split2sent(paragraph)
        Splits paragraph into sentences.

    """

    def __init__(self,json_sample):
        self.json_sample = json_sample
        self.par1 = json_sample["paragraph_1"]
        self.par2 = json_sample["paragraph_2"]
        self.to_align = True
        self.one2many = False
        self.correct = True
        self.alignment = []
        self.output_sentences = []

    def golden_answer(self):
        """
        Returns golden answer for the sample (if it is provided in the dataset)

    
        Raises
        ------
        KeyError
            if the sample does not contain golden answers or the sample is incorrectly formatted.

        Returns
        -------
        list
            List of tuples, where each tuple is a paraphrased sentence pair (i.e., one-to-one alignment)

        """
        ans = [(q["sen1"],q["sen2"]) for q in self.json_sample["sen_pairs"]]
        f_ans = [x for x in ans if len(x[0])==1 and len(x[1])==1]
        final = [(x[0][0],x[1][0]) for x in f_ans]
        return final



    def _split2sent(self,paragraph):
        """
        Splits paragraph into sentences.

        Parameters
        ----------
        paragraph : str
            Input paragraph.

        Returns
        -------
        list
            List of Sentence objects.

        """
        sentence_list = []
        #returns list of sentences
        doc = nlp(paragraph)
        for x in doc.sentences:
            sentence_start = min([n.start_char for n in x.words])
            sentence_end = max([n.end_char for n in x.words])
            sentence_string = paragraph[sentence_start:sentence_end].strip("'")
            sentence_list.append(Sentence(x,sentence_string))
        return sentence_list


    def preprocess(self):
        """
        Preprocess sample to check if it should be aligned and output sentences for each paragraph.

        Returns
        -------
        list
            List of tuples of form (sentence,sentence_tokens) for the first paragraph.
        list
            List of tuples of form (sentence,sentence_tokens) for the second paragraph

        """

        sent1 = self._split2sent(self.par1)
        sent2 = self._split2sent(self.par2)
        sent1_list = [(x.sentence,x.tokens) for x in sent1 if x.is_real_sentence()]
        sent2_list = [(x.sentence,x.tokens) for x in sent2 if x.is_real_sentence()]

        if len(sent1_list)==0 or len(sent2_list)==0:
            self.correct = False
        if len(sent1_list)==1 and len(sent2_list)==1:
            self.to_align = False
        elif sum([len(sent1_list)==1,len(sent2_list)==1])==1:
            self.one2many = True

        return sent1_list,sent2_list

    def _remove_identical(self,sentence_lists):
        """
        Remove identical sentences between the two paragraphs since we do not need to align them.

        Parameters
        ----------
        sentence_lists : list
            List with two lists, each containing sentences from one paragraph.


        Returns
        -------
        list
            Cleaned list of sentences for the first paragraph.
        list
            Cleaned list of sentences for the second paragraph.

        """
        identical = [x for x in sentence_lists[0] if x in sentence_lists[1]]
        sent1_clean = [x for x in sentence_lists[0] if x not in identical]
        sent2_clean = [x for x in sentence_lists[1] if x not in identical]
        return sent1_clean,sent2_clean
    
    def cosine_similarity_char(self,sen1, sen2):
        """
        Calculates ngram-based character cosine similariy.

        Parameters
        ----------
        c1 : list
            First list of correct answers.
        c2 : list
            Second list of correct answers.
        e1: list
            First list of errors.
        e2: list
            Second list of errors.
        sig_type: str
            Either "mcnemar" or "chi2", select method of calculation.

        Raises
        ------
        ValueError
            if "sig_type" is neither "mcnemar" or "chi2".

        Returns
        -------
        float
            calculated p-value.

        """
        ch1 = [ch for ch in sen1.lower()]
        ch2 = [ch for ch in sen2.lower()]
        a = ngrams(ch1, 3)
        b = ngrams(ch2, 3)
      

        #term frequencise
        tf_vec1 = Counter(a)
        tf_vec2 = Counter(b)
        #retrieve idf from the dictionary
        vec1 = {}
        vec2 = {}
        for k in tf_vec1:
          try:
            vec1[k] = tf_vec1[k]/len(tf_vec1.keys())*idf[k]
          except:
            vec1[k] = tf_vec1[k]/len(tf_vec1.keys())*idf_rand
        for k in tf_vec2:
          try:
            vec2[k] = tf_vec2[k]/len(tf_vec2.keys())*idf[k]
          except:
            vec2[k] = tf_vec2[k]/len(tf_vec2.keys())*idf_rand

          
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return (float(numerator) / denominator)
    
    def document_vector(self,word2vec_model, sent_tokens):
      # remove out-of-vocabulary words
      doc = [word.lower() for word in sent_tokens if word.isalpha()]
      token_embed = []
      for word in doc:
        try:
          em = word2vec_model[word] 
          token_embed.append(em)
        except:
          #print(word)
          pass
      if len(doc)>1:
        sent_embed = np.mean(token_embed, axis=0)
      else:
        sent_embed=[]
      return sent_embed
 


    def doc_vector_bert(self,sent_text):
        tokens, embeddings = emb4bert.extract_word_embeddings(sent_text, mode="sum", layers=[-1,-2,-3,-4])
        spec = ["[CLS]","[SEP]"]
        indices2skip = [tokens.index(t) for t in tokens if t in spec]
        embeddings = np.delete(embeddings,indices2skip,axis=0)

        if len(embeddings)>1:
            sent_embed = np.mean(embeddings, axis=0)
            
        else:
            sent_embed=[]
        return sent_embed

    def _similarity(self,sentence_pair,similarity_metric):
        #input: each pair has form ((s1,t1),(s2,t2)) output: (sx,sy,score)
        sim=similarity_metric
        sent1 = sentence_pair[0]
        sent2 = sentence_pair[1]
        sent1_text = sent1[0]
        sent2_text = sent2[0]
        sent1_tokens = sent1[1]
        sent2_tokens = sent2[1]
        if sim=="token":
            sim = len(set(sent1_tokens).intersection(sent2_tokens))/((len(sent1_tokens)+len(sent2_tokens)))
        elif sim == "cls":
            if len(sent1_tokens)==0 or len(sent2_tokens)==0:
                sim = 0
            else:
                sim=modelcls.get_score(sent1_text,sent2_text,metric="cosine")
        elif sim=="s_bert":
            emb1 = model_sent.encode(sent1_text)
            emb2 = model_sent.encode(sent2_text)
            sim = util.dot_score(emb1,emb2)
            #sim = (sim+31)/(301+31)#nli-dev
            #sim = (sim+52)/(229+52)#nli-test
            #sim = (sim+12)/(230+12)#alber-dev
            #sim = (sim+25)/(298+25)#albert-test
            #sim = (sim+2)/(22+2)#ml-dev
            sim = (sim+6)/(29+6)#ml-test
        elif sim=='bert_score':
            if len(sent1_tokens)==0 or len(sent2_tokens)==0:
                sim = 0
            else:
                metric.add(prediction = sent1_text,reference = sent2_text)
                sim = metric.compute(lang="en")["f1"][0]
        elif sim=='word2vec':
            emb1 = self.document_vector(model_w2v,sent1_tokens)
            emb2 = self.document_vector(model_w2v,sent2_tokens)
            if len(emb1)>0 and len(emb2)>0:
              sim = 1 - spatial.distance.cosine(emb1,emb2)
            else:
              sim = 0
        elif sim=="sent2vec":
            if len(sent1_tokens)==0 or len(sent2_tokens)==0:
                sim = 0
            else:
                pass
        elif sim=="glove":
            emb1 = self.document_vector(model_glove,sent1_tokens)
            emb2 = self.document_vector(model_glove,sent2_tokens)
            if len(emb1)>0 and len(emb2)>0:
              sim = 1 - spatial.distance.cosine(emb1,emb2)
            else:
              sim = 0
        elif sim=="bert_avg":
            emb1 = self.doc_vector_bert(sent1_text)
            emb2 = self.doc_vector_bert(sent2_text)
            if len(emb1)>0 and len(emb2)>0:
              sim = 1 - spatial.distance.cosine(emb1,emb2)
            else:
              sim = 0
        elif sim =="wnet":
            all_syn1 = []
            all_syn2 = []
            if len(sent1_tokens)==0 or len(sent2_tokens)==0:
                sim = 0
            else:
                for x in sent1_tokens:
                    all_syn1.extend([j.name() for i in wn.synsets(x) for j in i.lemmas()])
                for x in sent2_tokens:
                    all_syn2.extend([j.name() for i in wn.synsets(x) for j in i.lemmas()])
                if (len(all_syn1)+len(all_syn2))==0:
                    sim = 0
                else:
                    sim = len(set(all_syn1).intersection(all_syn2))/((len(all_syn1)+len(all_syn2)))
            
        elif sim== 'ngram':
            sim = self.cosine_similarity_char(sent1_text,sent2_text)
            
        else:
            pass
        return (sentence_pair[0][0],sentence_pair[1][0],sim)

    def _best_in_list(self,tuple_list):
        return sorted(tuple_list,key=lambda x: x[1],reverse=True)[0]

    def _select_best_uni(self,sentences_w_scores,threshold):
        sent_leftside = defaultdict(list)
        sent_rightside = defaultdict(list)
        for p in sentences_w_scores:
            sent_leftside[p[0]].append((p[1],p[2]))
            sent_rightside[p[1]].append((p[0],p[2]))
        leftside_pairings = []
        for k in sent_leftside.keys():
            best_match = sorted(sent_leftside[k],key=lambda x:x[1],reverse=True)[0]
            leftside_pairings.append((k,best_match[0],best_match[1]))
        aligned = [(x[0],x[1]) for x in leftside_pairings if x[2]>threshold]
        return aligned
    
    def _select_best_bi(self,sentences_w_scores,threshold):
        sent_leftside = defaultdict(list)
        sent_rightside = defaultdict(list)
        for p in sentences_w_scores:
            sent_leftside[p[0]].append((p[1],p[2]))
            sent_rightside[p[1]].append((p[0],p[2]))
        leftside_pairings = []
        rightside_pairings = []
        for k in sent_leftside.keys():
            best_match = sorted(sent_leftside[k],key=lambda x:x[1],reverse=True)[0]
            leftside_pairings.append((k,best_match[0],best_match[1]))
        for k in sent_rightside.keys():
            best_match = sorted(sent_rightside[k],key=lambda x:x[1],reverse=True)[0]
            rightside_pairings.append((best_match[0],k,best_match[1]))
        all_pairings = leftside_pairings+rightside_pairings
        bidirectional = list(set([x for x in all_pairings if all_pairings.count(x)==2]))
        aligned = [(x[0],x[1]) for x in bidirectional if x[2]>threshold]
        return aligned  
              
    def align(self):
        sent1_list,sent2_list = self.preprocess()#of the form [(sentence,tokens)]
        if self.one2many == True:
            return False
        if self.correct == False:
            return False
        if self.to_align == False:
            aligned = [(sent1_list[0][0],sent2_list[0][0])]
        else:
            sentence_scores = [self._similarity(pair) for pair in product(sent1_list,sent2_list)]
            if aligner_mode=="uni":
                aligned = self._select_best_uni(sentence_scores,threshold=THRESHOLD)
            if aligner_mode=="bi":
                aligned = self._select_best_bi(sentence_scores,threshold=THRESHOLD)
            if aligner_mode=="gale":
                prealigned = []
                for (i1, i2), (j1, j2) in reversed(list(self._select_gale(sent1_list,sent2_list))):
                    prealigned.append(([x[0] for x in sent1_list[i1:i2]], [x[0] for x in sent2_list[j1:j2]]))
                final_aligned = [(x[0][0],x[1][0]) for x in aligned if len(x[0])==1 and len(x[1])==1]
                aligned = [x for x in final_aligned if self._check_sim(x)]
        self.alignment.extend(aligned)   
        return True

    
    def automatic_answer(self,similarity_metric,aligner_mode):
        possible = self.align(similarity_metric,aligner_mode)
        if possible:
            return self.alignment
        else:
            return []

    def _final_filter(self,sentence_pair):
        if len(sentence_pair[0].split())>22 or len(sentence_pair[1].split())>22:
            return False
        if sentence_pair[0].lower() in sentence_pair[1].lower() or sentence_pair[1].lower() in sentence_pair[0].lower():
            return False
        intersection = set(sentence_pair[0].lower().split()).intersection(set(sentence_pair[1].lower().split()))
        diff1 = [x for x in sentence_pair[0].lower().split() if x not in intersection]
        diff2 = [x for x in sentence_pair[1].lower().split() if x not in intersection]
        if len(diff1)<1 and len(diff2)<1:    
            return False
        else:
            return True
        
class Dataset():
    def __init__(self,jsonl_file) -> None:
        self.json_data = []
        with open(jsonl_file) as f:
            for line in f:
                self.json_data.append(json.loads(line))

    def evaluate(self,*similarity_metrics,mode="uni"):
        sample_list = [Sample(s) for s in self.json_data]
            
        model_extracted = []
        golden = []
        for s in sample_list:
            model_extracted.extend(s.automatic_answer())
            golden.extend(s.golden_answer())
        print("MODEL",model_extracted)
        print("golden",len(golden))
        correct = [x for x in model_extracted if x in golden]
        extra = [x for x in model_extracted if x not in golden]
        #print("EXTRA",extra)
        #print("CORRECT",correct)
        #print("GOLDEN",golden)
        missed = [x for x in golden if x not in model_extracted]
        print("missed",missed)
        try:
            precision = len(correct)/len(model_extracted)
        except:
            precision = 0
        try:
            recall = len(correct)/len(golden)
        except:
            recall = 0
        try:
            f1_score = 2*(precision*recall)/(precision+recall)
        except:
            f1_score = 0
        print(precision,recall,f1_score)
        return precision,recall,f1_score

    def create_alignment_file(self,similarity_metric,mode,outfile):
        pass

### HELPER FUNCTIONS FOR ALIGNMENT ###

def count_beads(json_sample_list):
    """
    Calculates beads for gale-church algorithm.

    Parameters
    ----------
    json_sample_list : list
        List of json samples of the form {"sen1":"","sen2":""}

    Raises
    ------
    ValueError
        if "ngram_size" outside of [1,10] range.
    TypeError
        if "ngram_size" is not an integer.

    Returns
    -------
    Counter
        counter with sentence lenghts.
    """
    c = Counter()
    for s in json_sample_list:
        sen_list = s["sen_pairs"]
        for p in sen_list:
            i = len(p["sen1"])
            j = len(p["sen2"])
            c[(i,j)]+=1
    return c

def get_idf(json_sample_list,ngram_size=1):
    """
    Calculates idf for a list of json samples.

    Parameters
    ----------
    json_sample_list : list
        List of json samples of the form {"paragraph_1":"","paragraph_2":""}
    ngram_size : int
        Size of the ngram to use. Must be an integer from 1 to 10.
  
    Raises
    ------
    ValueError
        if "ngram_size" outside of [1,10] range.
    TypeError
        if "ngram_size" is not an integer.

    Returns
    -------
    dict
        calculated idf value for each ngram.
    float
        calculated for all input text.

    """
    if not isinstance(ngram_size, int):
         raise TypeError("Ngram size must be an integer.")
    if ngram_size not in range(1,10):
         raise ValueError("Ngram size outside of reasonable range.")

    all_text = [s["paragraph_1"] for s in json_sample_list]+[s["paragraph_2"] for s in json_sample_list]
    a = [list(ngrams(doc.lower(), ngram_size)) for doc in all_text]
    all_ngram = list(set([x for listx in a for x in listx]))
    c = Counter()
    for ngram in all_ngram:
        for doc in a:
            if ngram in doc:
                c[ngram]+=1
    idf = {}
    for k in c.keys():
        idf[k] = math.log((len(all_text)+1)/c[k]+1)
    idf_rand = math.log((len(all_text)+1)/1)
    return idf,idf_rand

### ANALYSIS AND STATISTICS ###

def check_significance(c1,c2,e1,e2,sig_type="mcnemar"):
    """
    Calculates p_value using eaither chi^2 or McNemar test. McNemar is preferred.

    Parameters
    ----------
    c1 : list
        First list of correct answers.
    c2 : list
        Second list of correct answers.
    e1: list
        First list of errors.
    e2: list
        Second list of errors.
    sig_type: str
        Either "mcnemar" or "chi2", select method of calculation.

    Raises
    ------
    ValueError
        if "sig_type" is neither "mcnemar" or "chi2".

    Returns
    -------
    float
        calculated p-value.

    """
    if sig_type not in ["mcnemar","chi2"]:
        raise ValueError("Incorrect significance testing method")
    inc1cor2 = len([x for x in e1 if x not in e2])
    cor1inc2 = len([x for x in e2 if x not in e1])
    cor1cor2 = len(set(c1).intersection(set(c2)))
    inc1inc2 = len(set(e1).intersection(set(e2)))
    if sig_type=="chi2":
        a = np.array([[cor1cor2,cor1inc2],[inc1cor2,inc1inc2]])
        x2_statistic = (np.absolute(a[0, 1] - a[1, 0]) - 1) ** 2 / (a[0, 1] + a[1, 0])
        p_value = chi2.sf(x2_statistic, 1)
    else:
        p_value = mcnemar(a)
    return p_value


if __name__ == "__main__":
    pass

