import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
import nltk


nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
#1-getting data
data=pd.read_csv("ld.csv")


#2-cleaning the data
data=data.drop(["is_premium","discuss_count","frequency","title","solution_link","companies","url","likes","dislikes","accepted","similar_questions","submissions","asked_by_faang"],axis=1)


#print(data.isnull().sum())
#254 rows found with null values
data=data.dropna()

#3-data processing
num_list=["second","third","two","three","four","fourth","five","fifth","six","sixth"]
def hasArray(s):
    if('array' in s):
        return 1
    else:
        return 0
def has2Array(s):
    if('two array' in s):
        return 1
    else:
        return 0
def hasString(s):
    if('string' in s):
        return 1
    else:
        return 0
def has2String(s):
    if('two string' in s):
        return 1
    else:
        return 0
def lem_and_to_lower(s):
    s=s.lower()
    li=list(s.split(" "))
    s2=""
    for w in li:
        s2=s2+" "+wnl.lemmatize(w)
    return s2
    
def hasReverse(s):
    if('reverse in s'):
        return 1
    else:
        return 0

def hasSort(s):
    if "sort" in s:
        return 1
    else:
        return 0
def hasSS(s):
    if "substring" in s:
        return 1
    else:
        return 0
def hasRep(s):
    if "without repeat" in s:
        return 0
    else:
        return 1

def isAscend(s):
    if("non decrease" or "non-decrease" or "ascend" in s):
        return 1
    else:
        return 0
def isMin(s):
    if "minimize" in s:
        return 1
    else:
        return 0
def isMin2(s):
    if "minimum" in s:
        return 1
    else:
        return 0
def hasPrime(s):
    if "prime" in s:
        return 1
    else:
        return 0
def hasDuplicates(s):
    if "duplicate" in s:
        return 1
    else:
        return 0
def isMatrix(s):
    if "matrix" in s:
        return 1
    else:
        return 0
def isRotate(s):
    if "rotate" in s:
        return 1
    else:
        return 0
def isGrid(s):
    if "grid" in s:
        return 1
    else:
        return 0
def isTree(s):
    if "tree" in s:
        return 1
    else:
        return 0
def isGraph(s):
    if "graph" in s:
        return 1
    else:
        return 0
def isTree2(s):
    if "tree" and ("height" or "depth") in s:
        return 1
    else:
        return 0
def isTree3(s):
    if "tree" and "zigzag" in s:
        return 1
    else:
        return 0
def isTree4(s):
    if "tree" and "maximum depth" in s:
        return 1
    else:
        return 0
def isTree5(s):
    if "tree" and "minimum depth" in s:
        return 1
    else:
        return 0
def hasConsecutive(s):
    if("consecutive" in s):
        return 1
    else:
        return 0
def hasPalindrome(s):
    if("palindrome" in s):
        return 1
    else:
        return 0
def use_heap(s,x):
    flag=0
    for y in num_list:
        if y or "k" in s and x in s:
            flag=1
            break
    return flag
data=pd.get_dummies(data,columns=["difficulty"])
data["desc"]=data["description"].apply(lambda i:lem_and_to_lower(i))
data['has_array']=data['desc'].apply(lambda i:hasArray(i))
data['has2_array']=data['desc'].apply(lambda i:has2Array(i))
data['has_string']=data['desc'].apply(lambda i:hasString(i))
data['has2_string']=data['desc'].apply(lambda i:has2String(i))
data['has_reverse']=data['desc'].apply(lambda i:hasReverse(i))


data=data.drop(["description"],axis=1)
data['sorted']=data['desc'].apply(lambda i:hasSort(i))
data['substring']=data['desc'].apply(lambda i:hasSS(i))
data['repeating']=data['desc'].apply(lambda i:hasRep(i))
data["ascending"]=data['desc'].apply(lambda i:isAscend(i))
data["minimize"]=data['desc'].apply(lambda i:isMin(i))

data["prime"]=data['desc'].apply(lambda i:hasPrime(i))
data["duplicate"]=data['desc'].apply(lambda i:hasDuplicates(i))
data["matrix"]=data['desc'].apply(lambda i:isMatrix(i))
data["rotate"]=data['desc'].apply(lambda i:isRotate(i))
data["grid"]=data['desc'].apply(lambda i:isGrid(i))
data["minimum"]=data['desc'].apply(lambda i:isMin2(i))

data["tree"]=data["desc"].apply(lambda i:isTree(i))
data["tree_height"]=data["desc"].apply(lambda i:isTree2(i))
data["tree_zigzag"]=data["desc"].apply(lambda i:isTree3(i))
data["tree_depth"]=data["desc"].apply(lambda i:isTree4(i))
data["tree_depth2"]=data["desc"].apply(lambda i:isTree5(i))

data['consecutive']=data['desc'].apply(lambda i:hasConsecutive(i))
data['palindrome']=data['desc'].apply(lambda i:hasPalindrome(i))

data["graph"]=data["desc"].apply(lambda i:isGraph(i))

def forAll(s,x):
    if x in s:
        return 1
    else:
        return 0
data["cycle"]=data["desc"].apply(lambda i:forAll(i,"cycle"))
data["pair"]=data["desc"].apply(lambda i:forAll(i,"pair"))
data["arrange"]=data["desc"].apply(lambda i:forAll(i,"arrange"))
data["bit"]=data["desc"].apply(lambda i:forAll(i,"bit"))
data["house"]=data["desc"].apply(lambda i:forAll(i,"house"))
data["head"]=data["desc"].apply(lambda i:forAll(i,"head"))
data["contigous"]=data["desc"].apply(lambda i:forAll(i,"contigous"))
data["permutation"]=data["desc"].apply(lambda i:forAll(i,"permutation"))
data["combination"]=data["desc"].apply(lambda i:forAll(i,"combination"))
data["node"]=data["desc"].apply(lambda i:forAll(i,"node"))
data["unique"]=data["desc"].apply(lambda i:forAll(i,"unique"))
data["lca"]=data["desc"].apply(lambda i:forAll(i,"lca"))
data["delete"]=data["desc"].apply(lambda i:forAll(i,"delete"))
data["search"]=data["desc"].apply(lambda i:forAll(i,"search"))
data["target"]=data["desc"].apply(lambda i:forAll(i,"target"))
data["game"]=data["desc"].apply(lambda i:forAll(i,"game"))
data["distinct"]=data["desc"].apply(lambda i:forAll(i,"distinct"))
data["perfect_square"]=data["desc"].apply(lambda i:forAll(i,"perfect square"))
data["range"]=data["desc"].apply(lambda i:forAll(i,"range"))
data["sorted_matrix"]=data["desc"].apply(lambda i:forAll(i,"sort matrix"))
data["first"]=data["desc"].apply(lambda i:forAll(i,"first"))
data["last"]=data["desc"].apply(lambda i:forAll(i,"last"))
data["difference"]=data["desc"].apply(lambda i:forAll(i,"difference"))
data["sum"]=data["desc"].apply(lambda i:forAll(i,"sum"))
data["repeat"]=data["desc"].apply(lambda i:forAll(i,"repeat"))
data["remove"]=data["desc"].apply(lambda i:forAll(i,"remove"))
data["jump"]=data["desc"].apply(lambda i:forAll(i,"jump"))


data["k_max"]=data["desc"].apply(lambda i:use_heap(i,"maximum"))
data["k_min"]=data["desc"].apply(lambda i:use_heap(i,"minimum"))
data["k_max2"]=data["desc"].apply(lambda i:use_heap(i,"big"))
data["k_min2"]=data["desc"].apply(lambda i:use_heap(i,"small"))

data["xor"]=data["desc"].apply(lambda i:forAll(i,"xor"))
data["segment"]=data["desc"].apply(lambda i:forAll(i,"segment"))
data["count"]=data["desc"].apply(lambda i:forAll(i,"count"))




y=data.iloc[:,2]
X=data.iloc[:,4:]

st=set()
for i,j in y.iteritems():
    L=(j.split(","))
    for l in L:
        st.add(l)

st=list(st)
#filling colum names in Y
Y=pd.DataFrame(columns=st)


#filling rows names in Y
for i,j in y.iteritems():
    
    for c in Y.columns:
        if(c in j):
            Y.at[i, c] = 1
        else:
            Y.at[i, c] = 0

X=X.drop(["desc"],axis=1)

shape_x=X.shape
# 1571,61
shape_y=Y.shape
# 1571,43




X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42,test_size=0.20,shuffle=True)
ytest_shape=Y_test.shape
# 315,43
xtest_shape=X_test.shape
# 315,61
ytrain_shape=Y_train.shape
# 1256,43
xtrain_shape=X_train.shape
# 1256,61
for c in X_train.columns:
    X_train[c]=X_train[c].astype(float)
for c in Y_train.columns:
    Y_train[c]=Y_train[c].astype(float)
for c in X_test.columns:
    X_test[c]=X_test[c].astype(float)
for c in Y_test.columns:
    Y_test[c]=Y_test[c].astype(float)


dt=DecisionTreeClassifier()
moc=MultiOutputClassifier(dt)
moc.fit(X_train,Y_train)

