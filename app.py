from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn
import io
import requests
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


from rivescript import RiveScript
import os.path
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
	

	# In[201]:


	dataset = pd.read_csv("graddata2.csv")
	print(type(dataset))


	# In[202]:


	#looking into head
	print(dataset.head())

	#no of rows, cols
	print(dataset.shape)

	#info bout emm data
	print(dataset.info())


	# In[203]:


	print(dataset.sum())


	# In[204]:


	print(dataset.describe())


	# In[205]:


	#get mean
	print(dataset.mean())


	# In[206]:


	#get median
	print(dataset.median())




	# In[208]:


	#convert dataframe into matrix
	dataArray = dataset.values

	#splitting input features & o/p vars
	X = dataArray[:,1:8]
	y = dataArray[:,0:1]


	# In[209]:


	#splitting training & testing
	validation_size = 0.10
	seed = 9
	X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=validation_size, random_state = seed)


	# In[210]:


	#create prediction model
	model = LogisticRegression()
	#fit model
	model.fit(X_train, Y_train)
	#predict!
	predictions = model.predict(X_test)


	# In[211]:


	print("Model --- LogisticRegression")
	print("Accuracy: {} ".format(accuracy_score(Y_test,predictions) * 100))
	print(classification_report(Y_test, predictions))


	# In[225]:


	new_data = [(330,9,115,2,1,4,4), (317,8.6,95,5,1,5,5), (317,8.6,95,3,1,5,5)]
	new_array = np.asarray(new_data)
	labels=["reject","admit"]


	# In [226]:
	new_data = [(330,9,115,2,1,4,4)]
	new_array = np.asarray(new_data)
	labels=["reject","admit"]

	# In[227]:


	prediction=model.predict(new_array)
	#get no of test cases used
	no_of_test_cases, cols = new_array.shape

	#res
	for i in range(no_of_test_cases):
		print("Status of Student with GRE scores = {}, GPA grade = {}, Toefl = {} , Rank = {}, Reseacrh = {}, SOP = {}, LOR = {} will be ----- {}".format(new_data[i][0],new_data[i][1],new_data[i][2],new_data[i][3],new_data[i][4],new_data[i][5],new_data[i][6], labels[int(prediction[i])]))


	# In[250]:

	

	file = os.path.dirname("__file__")
	convo = os.path.join(file , 'convo')
	
	bot = RiveScript()
	bot.load_directory(convo)
	bot.sort_replies()
	while True:
		userText = request.args.get('msg')
		msg = userText
		#print(msg)
		if msg.find('gre') != -1 :
			global gre
			gre = [int(i) for i in msg.split() if i.isdigit()] 
			#print (gre[0])
		if msg.find('cgpa') != -1 :
			global cgpa
			cgpa = re.findall("\d+\.\d+", msg)
			cgpa = [float(i) for i in cgpa]
			#print (cgpa[0])
			
		if msg.find('toefl') != -1 :
			global toefl
			toefl = [int(i) for i in msg.split() if i.isdigit()] 
			
		
		if msg.find('rank') != -1 :
			global uniRank
			uniRank = [int(i) for i in msg.split() if i.isdigit()] 
			
			
		if msg.find('sop') != -1 :
			global sop
			sop = re.findall("\d+\.\d+", msg)
			sop = [float(i) for i in sop] 
			
			
		if msg.find('lor') != -1 :
			global lor
			lor = re.findall("\d+\.\d+", msg)
			lor = [float(i) for i in lor]
			
		
		if msg.find('research') != -1 :
			global research
			if msg.find('yes')!= -1:
				research = 1
			else:
				research = 0
		
		if msg.find('find') != -1 :
			new_data = [(gre[0],cgpa[0],toefl[0],uniRank[0],research,sop[0],lor[0])]
			new_array = np.asarray(new_data)
		
			labels=["reject","admit"]
			
			prediction=model.predict(new_array)
			
			botText= labels[int(prediction) ]+'. Do you want to start again?'
			
			
			return str(botText)
		botText=bot.reply('localuser',msg)
		
		return str(botText)
		if msg == 'bye':
			break


if __name__ == "__main__":
    app.run(debug=True)