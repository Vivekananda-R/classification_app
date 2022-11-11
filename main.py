import sklearn.datasets as datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import streamlit as st

# ------------------Main page----------------------------
st.set_page_config(page_icon=":smile:",page_title="Classifier",layout="centered",initial_sidebar_state = "expanded")
st.title(" Classifier ")
st.subheader(""" Explore different classifier which one is best?""")
dataset = st.sidebar.selectbox("Select a dataset",["Iris","Wine Dataset","Breast Cancer"])
classifier = st.sidebar.selectbox("Select a Classifier",["KNN","SVM","Random Forest"])



# --------------------------Functions---------------------------
def get_dataset(datas):
    if datas == "Iris":
        data = datasets.load_iris()
    elif datas == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    X = data.data 
    y = data.target 
    return X ,y

def add_parameter(classifier):
    params = dict()
    if classifier == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif classifier == "SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        depth = st.sidebar.slider("Max depth ",2,15)
        no_esti = st.sidebar.slider("No.of Estimator",1,100)
        params["max_depth"] = depth
        params["no_esti"] = no_esti
    return params

def get_classifier(classifier,params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier( n_estimators=params["no_esti"],
                                     max_depth=params["max_depth"],
                                     random_state=1234 )
    return clf
        
# ------------------Calling functions -------------------------------- 
X,y  = get_dataset(dataset)  # returning values based on data set
f"##### Dataset being used is `{dataset} Dataset`"
col1,col2 = st.columns((2,2))
col1.write(f"Shape of datasets {X.shape}") 
col2.write(f"Number of classes are {len(np.unique(y))}")

param = add_parameter(classifier) #parameter functions based on classifier selected

clf = get_classifier(classifier,param)

# ------------------Classification --------------------------------
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state = 1234)

clf.fit(xtrain,ytrain)
y_predict = clf.predict(xtest)
acc = accuracy_score(ytest,y_predict)

col1.write("### Classifier = `{}`".format(classifier))
col2.write("### Accuracy `{:.4f}`".format(acc))


#----------------Plottings --------------------------------
pca = PCA(2) #feature reduction algoritnm for displaying classifier results
xprojected = pca.fit_transform(X)

x1 = xprojected[:,0]
x2 = xprojected[:,1]

fig = px.scatter(x1,x2
                 ,color=y
                 ,color_discrete_sequence=["lightblue", "green", "yellow","pink"]
                 ,hover_data=[0]
                 ,labels = dict(x='X-axis',index = "Y-axis")
                 )
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
st.plotly_chart(fig,use_container_width=True)