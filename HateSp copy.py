import re

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag, pos_tag_sents, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#appling model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#nltk.download()
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('stopwords')
dataset=pd.read_csv('labeled_data.csv')
data = dataset.loc[:,['class','tweet']]
#Dataset Cleaning
data['tweet'].replace(regex=r'@[\w]* ?:? ?',value='',inplace=True)
data['tweet'].replace(regex=r'https?://[\w./]*',value='',inplace=True)  
data['tweet'].replace(regex=r'&[^\s]*;',value='',inplace=True)
data['tweet'].replace(regex=r'n\'t',value=' not',inplace=True)
data['tweet'].replace(regex=r'[ ]?[^a-z]?[R]{1}[T]{1}[ ]+',value=' ',inplace=True)
#function for creating postagging
#creating df with tweet,postag,class
df = pd.DataFrame(data['tweet'])
texts = df['tweet'].tolist()
tokenized_texts = pos_tag_sents(map(word_tokenize, texts))
df['POS'] = tokenized_texts
df['class'] = data['class']
  
#1. Sentimental features
#write adjectives,adverbs and verbs in pos_tag.csv                   
df_len = df.shape[0] 
file = open("pos_tag.csv","a")

'''
for row in range(0,df_len):
    line=""
    tags =df.loc[row,'POS']
    tags_len = len(tags)
    for i in range(0,tags_len):
        if str(tags[i][1]).find('JJ') == 0 or str(tags[i][1]).find('RB') == 0 or str(tags[i][1]).find('VB') == 0:
            line += str(tags[i][0])+" "
    line+="\n"
    file.write(line)
'''
    #print(df.loc[row,'tweet'])
#call java code to calculate sentiment scores and write them in pos_tag.csv
#working on pos_tag.csv modified in java
negation_list = ['not','never','neither','nor','no','nowhere','nothing']
contrast_list = ['but','however','even though','although','despite','in spite of','while','whereas','unlike']
file = open("pos_tag.csv","r")
tweet_score = pd.DataFrame(columns = ['pw','nw','pos_neg_ratio'])
tweet_word_score = list()
for index,row in df.iterrows():
    tweet = str(row['tweet']).split()
    line = str(file.readline()).split()
    pw_sum = 0
    nw_sum= 0
    pnr=0
    negation_occurred = 0
    contrast_occurred = 0
    pos_list = ['JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
    for word in tweet:
        neg_val = 1
        if word in negation_list:
            negation_occurred=1
            contrast_occurred=0
        elif word in contrast_list:
            negation_occurred=0
            contrast_occurred=1
        elif negation_occurred==1 and contrast_occurred==0:
            for i in row['POS']:
                if i[1] in pos_list:
                   neg_val=-1 
        w=re.sub("[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~0-9]+[a-zA-Z]*","",word)
        if w in line:
            score = int(line[line.index(w)-1])*neg_val
            line[line.index(w)-1] = score
            if score<0:
                nw_sum+=score
            elif score>0:
                pw_sum+=score
                
        if (pw_sum+nw_sum) ==0:
            pnr = 0 
        else:
            pnr=(pw_sum+nw_sum)/(pw_sum-nw_sum)
    
    tweet_score=tweet_score.append({'pw':pw_sum,'nw':nw_sum,'pos_neg_ratio':pnr},ignore_index=True)
    if len(line)==0:
        tweet_word_score.append(0)
    else:
        tweet_word_score.append(line)
#End Sentimental features
#2.semantic features
cap=[]
sementic_df=pd.DataFrame(columns=['excp','que','dot','caps','qut','intj','nwrd'])
for index,row in df.iterrows():
     tweet=str(row['tweet'])
     excp=0
     que=0
     dot=0
     caps=0
     qut=0
     intj=0
     nwrd=0
     excp=tweet.count("!",0,len(tweet))
     que=tweet.count("?",0,len(tweet))
     dot=tweet.count(".",0,len(tweet))
     qut=(tweet.count("\"",0,len(tweet))+tweet.count("\'",0,len(tweet)))
     t=tweet
     t=re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~0-9]',' ',t)
     nwrd=len(t.split())
     t=tweet
     t=re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~0-9]+','',t)
     t=re.sub(r' [a-z]+ ',' ',t)
     t=re.sub(r' [A-Z]+[a-z]+',' ',t)
     t=re.sub(r' [a-z]+[A-Z]+',' ',t)
     t=re.sub(r'[A-Z]+[a-z]+',' ',t)
     t=re.sub(r' [a-zA-Z]+[a-z]+[A-Za-z]',' ',t)
     t=re.sub(r'[a-z]+',' ',t)
     t=re.sub(r' [I]{1} ',' ',t)
     caps=len(t.split())
     cap.append(t.split())    
     t = row['POS']
     for i in t:
         if i[1]=="UH":
            intj+=1  
     sementic_df = sementic_df.append({'excp':excp,'que':que,'dot':dot,'caps':caps,'qut':qut,'intj':intj,'nwrd':nwrd}, ignore_index=True)   
#End of semantic features 
#Dataset cleaning (Stopwords ,punctuations,#tags)
df['tweet'].replace(regex=r'[!"#$%&\'()*+,-./:;<=>?@0-9[\]^_`{|}~]+',value=' ',inplace=True)
df['tweet'].replace(regex=r'[#][a-zA-Z0-9]+',value='',inplace=True)
stop = stopwords.words('english')
df['tweet'] = df['tweet'].apply(lambda x:' '.join([word.lower() for word in x.split() if word.lower() not in stop]))
texts = df['tweet'].tolist()
tokenized_texts = pos_tag_sents(map(word_tokenize, texts))
df['POS'] = tokenized_texts
#3. Unigram features
pos_list = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
#creating 3 dictionaries
hate_count=dict();
offensive_count=dict();
clean_count=dict();
#iterate df
for index,row in df.iterrows():
    tweet =row['POS']
    for word in tweet:
        if word[1] in pos_list:
            if row['class']==0:
                hate_count[word[0]] = hate_count.get(word[0],0) + 1
            elif row['class']==1:
                offensive_count[word[0]] = offensive_count.get(word[0],0) + 1
            elif row['class']==2:
                clean_count[word[0]] = clean_count.get(word[0],0) + 1

ratio = pd.DataFrame(columns=['word','h1','h2','o1','o2','c1','c2'])
for key,value in hate_count.items():
    if offensive_count.get(key,0)==0:
        h1 = 2
    else:
        h1 = value/offensive_count[key]
        
    if clean_count.get(key,0)==0:
        h2 = 2
    else:
        h2 = value/clean_count[key]
    ratio = ratio.append({'word':key,'h1':h1,'h2':h2,'o1':0,'o2':0,'c1':0,'c2':0},ignore_index=True)
   
#offensive_ratio = pd.DataFrame(columns=['word','r1','r2'])
for key,value in offensive_count.items():
    if hate_count.get(key,0)==0:
        o1 = 2
    else:
        o1 = value/hate_count[key]
        
    if clean_count.get(key,0)==0:
        o2 = 2
    else:
        o2 = value/clean_count[key]
    search = ratio.index[ratio['word']==key].tolist()
    if len(search)==1:
        ratio.loc[search[0],'o1'] = o1
        ratio.loc[search[0],'o2'] = o2
    else:
        ratio.append({'word':key,'h1':0,'h2':0,'o1':o1,'o2':o2,'c1':0,'c2':0},ignore_index=True)
 

for key,value in clean_count.items():
    if hate_count.get(key,0)==0:
        c1 = 2
    else:
        c1 = value/hate_count[key]
        
    if offensive_count.get(key,0)==0:
        c2 = 2
    else:
        c2 = value/offensive_count[key]
    search = ratio.index[ratio['word']==key].tolist()
    if len(search)==1:
        ratio.loc[search[0],'c1'] = c1
        ratio.loc[search[0],'c2'] = c2
    else:
        ratio.append({'word':key,'h1':0,'h2':0,'o1':0,'o2':0,'c1':c1,'c2':c2},ignore_index=True)  
#End Unigram features  
#4.pattern features
hate_pattern_count = {}
offensive_pattern_count = {}
clean_pattern_count = {}
sentimental_word_pattern=[]
for index,row in df.iterrows():    
     tweet_with_pos=row['POS']
     sw_pattern=""
     for t in tweet_with_pos:
         tws = tweet_word_score[index]
         if tws!=0:
             if t[0] in tws:#checking if word has senti score or not
                 word_index = tws.index(t[0]) 
                 if int(tws[word_index-1])<0:
                     sw_pattern+= "Negative_"
         if t[1] in ['NNPS','NNP','NNS','NN']:
             sw_pattern+="NOUN "
         elif t[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
             sw_pattern+="VERB "
         elif t[1] in ['RB','RBR','RBS']:   
             sw_pattern+="ADVERB "
         elif t[1] in ['JJ','JJR','JJS']:   
             sw_pattern+="ADJECTIVE "
     sentimental_word_pattern.append(sw_pattern)     
     if row['class']==0:
                hate_pattern_count[sw_pattern] = hate_pattern_count.get(sw_pattern,0) + 1
     elif row['class']==1:
                offensive_pattern_count[sw_pattern] = offensive_pattern_count.get(sw_pattern,0) + 1        
     elif row['class']==2:
                clean_pattern_count[sw_pattern] = clean_pattern_count.get(sw_pattern,0) + 1        

pattern_ratio = pd.DataFrame(columns=['pattern','h1','h2','o1','o2','c1','c2'])
for key,value in hate_pattern_count.items():
    if offensive_pattern_count.get(key,0)==0:
        h1 = 2
    else:
        h1 = value/offensive_pattern_count[key]
        
    if clean_pattern_count.get(key,0)==0:
        h2 = 2
    else:
        h2 = value/clean_pattern_count[key]
    pattern_ratio = pattern_ratio.append({'pattern':key,'h1':h1,'h2':h2,'o1':0,'o2':0,'c1':0,'c2':0},ignore_index=True)
   
#offensive_ratio = pd.DataFrame(columns=['word','r1','r2'])
for key,value in offensive_pattern_count.items():
    if hate_pattern_count.get(key,0)==0:
        o1 = 2
    else:
        o1 = value/hate_pattern_count[key]
        
    if clean_pattern_count.get(key,0)==0:
        o2 = 2
    else:
        o2 = value/clean_pattern_count[key]
    search = pattern_ratio.index[pattern_ratio['pattern']==key].tolist()
    if len(search)==1:
        pattern_ratio.loc[search[0],'o1'] = o1
        pattern_ratio.loc[search[0],'o2'] = o2
    else:
        pattern_ratio.append({'pattern':key,'h1':0,'h2':0,'o1':o1,'o2':o2,'c1':0,'c2':0},ignore_index=True)
 
for key,value in clean_pattern_count.items():
    if hate_pattern_count.get(key,0)==0:
        c1 = 2
    else:
        c1 = value/hate_pattern_count[key]
        
    if offensive_pattern_count.get(key,0)==0:
        c2 = 2
    else:
        c2 = value/offensive_pattern_count[key]
    search = pattern_ratio.index[pattern_ratio['pattern']==key].tolist()
    if len(search)==1:
        pattern_ratio.loc[search[0],'c1'] = c1
        pattern_ratio.loc[search[0],'c2'] = c2
    else:
        pattern_ratio.append({'pattern':key,'h1':0,'h2':0,'o1':0,'o2':0,'c1':c1,'c2':c2},ignore_index=True)  
#End pattern features       

#extracting unigram features above threshold value 
TH_Unigram= 0.12
ratio_min = ratio[(ratio['h1']>TH_Unigram) & (ratio['h2']>TH_Unigram) & (ratio['o1']>TH_Unigram) & (ratio['o2']>TH_Unigram) & (ratio['c1']>TH_Unigram) & (ratio['c2']>TH_Unigram)]
vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(df['tweet'].tolist())         
extracted_unigrams = ratio_min['word'].tolist() 
#names = list(vectorizer.get_feature_names())   
df_unigrams = pd.DataFrame(columns=['tweet'])
df_unigrams['tweet'] = df['tweet'].apply(lambda x:' '.join([word for word in x.split() if word in extracted_unigrams]))
X = vectorizer.fit_transform(df_unigrams['tweet'].tolist()) 
names = list(vectorizer.get_feature_names()) 


#creating pattern feature matrix     
pattern_ratio['pattern'].replace(regex=r' ',value='_',inplace=True)
sentimental_word_pattern[:] = [s.replace(' ', '_') for s in sentimental_word_pattern]
df['pattern'] = sentimental_word_pattern
#extracting pattern features above threshold value
TH_Pattern=0.12
pattern_ratio_min = pattern_ratio[(pattern_ratio['h1']>TH_Pattern) & (pattern_ratio['h2']>TH_Pattern) & (pattern_ratio['o1']>TH_Pattern) & (pattern_ratio['o2']>TH_Pattern) & (pattern_ratio['c1']>TH_Pattern) & (pattern_ratio['c2']>TH_Pattern)]
extracted_pattern = pattern_ratio_min['pattern'].tolist() 
vectorizer = CountVectorizer()        
df_pattern = pd.DataFrame(columns=['pattern'])
df_pattern['pattern'] = df['pattern'].apply(lambda x:' '.join([word for word in x.split() if word in extracted_pattern]))
Y = vectorizer.fit_transform(df_pattern['pattern'].tolist()) 
pattern_names = list(vectorizer.get_feature_names())


#combine symantic ,sentimental and unigram features 
pn_ratio = np.array(tweet_score['pos_neg_ratio'].tolist()).reshape(-1, 1)
all_feature=np.append(np.asarray(X.todense()),pn_ratio,axis=1)
sementic_fea = np.array(sementic_df.values.tolist()).reshape(-1, 7)
all_feature=np.append(all_feature,sementic_fea,axis=1) 
#combine pattern features
all_feature=np.append(all_feature,np.asarray(Y.todense()),axis=1)

all_features=sparse.csr_matrix(all_feature) 


X_train, X_test, y_train, y_test = train_test_split(all_features, df['class'], test_size = 0.20)



svclassifier = SVC(kernel='rbf')  

svclassifier.fit(X_train, y_train)  
y_pred = svclassifier.predict(X_test) 
#cheaking accuracy
y_testa=y_test.values
acc=0  
len_test = len(y_test)
for i in range(0,len_test):
    if y_pred[i]==y_testa[i]:    
        acc+=1
print(acc/len_test)      
