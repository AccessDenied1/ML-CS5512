
from nltk import word_tokenize
from nltk import download
import numpy as np
np.set_printoptions(suppress=True)
from hmmlearn.hmm import MultinomialHMM

data = open("hmm-train.txt", "r").read()
#Task 1 cleaning the data
download('punkt')
def preprocess(text):
    doc = text.upper()
    doc = word_tokenize(doc)
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    D = " ".join(doc)
    return D
data = preprocess(data) # This is our preprocessed data
print("Task - 2")
print("So the two states are vowel and consonant")
print("0 -> Vowel & 1-> Consonant")

trans = np.zeros((2,2)) 
emmision = np.zeros((27,2))
vowel = "AEIOUY "
trans_prob = np.zeros((2,2))
emmis_prob = np.zeros((27,2))
pre_state = 1
idx = ord('T') - 65 # The first charcter ('T') in the data
emmision[idx][pre_state] = 1

#This will calculate the transition and emmision probs
k = 0
for i in data:
    if(k == 0):
        k = k+1
        continue
    if i is " ":
        idx = 26
    else:
        idx = ord(i) - 65
    if i in vowel:
        state = 0
        trans[pre_state][state] = trans[pre_state][state]+1
        pre_state = 0
    else:
        state = 1
        trans[pre_state][state] = trans[pre_state][state]+1
        pre_state = 1
    emmision[idx][pre_state] = emmision[idx][pre_state] + 1
    
trans_prob[0] = np.divide(trans[0],sum(trans[0]))
trans_prob[1] = np.divide(trans[1],sum(trans[1]))

print("The transition prob of this natural model : ")
print(trans_prob)

emmis_prob[:,0] = np.divide(emmision[:,0],sum(emmision[:,0]))
emmis_prob[:,1] = np.divide(emmision[:,1],sum(emmision[:,1]))
print("\nThe emmision prob of this natural model : ")
print("  State-0    State-1")
print(emmis_prob)
def seven_most_probabe(emmis_prob):
    al = emmis_prob[:,0].argsort()[-7:][::-1]
    al2 = emmis_prob[:,1].argsort()[-7:][::-1]
    print("\n7 most likely characters in vowels")
    for i in al:
        if i == 26:
            print("\tSpace")
        else:
            print("\t",chr(i+65))
    print("\n7 most likely characters in consonants")
    for i in al2:
        if i == 26:
            print("\tSpace")
        else:
            print("\t",chr(i+65))
seven_most_probabe(emmis_prob) #printing the 7 most likely characters
print("\nTask - 3")

def convert(string): # mapping function, map A->0 , B->1 , C->3 ... 
    output = []
    for character in string:
        if character is " ":
            number = 26
        else:
            number = ord(character) - 65
        output.append(number)
    return (output)
data2 = convert(data) #Convert the data from stream of chacters to stream of numbers
DD = np.array(data2)
Data_arr = DD.reshape((DD.shape[0],1))
model = MultinomialHMM(n_components=2,n_iter=200, tol=0.01, verbose=False)
print("Training started")
model.fit(Data_arr)
print("Training Done")
print("Model = ",model.monitor_)
print("The transition prob of this trained model : ")
print(model.transmat_)
emiso = np.transpose(model.emissionprob_)
print("\nThe emmision prob of this trained model : ")
print("  State-0    State-1")
print(emiso)
seven_most_probabe(emiso) #printing the 7 most likely characters
print("Stationary probbabilities : ",model. get_stationary_distribution())
print("So seeing the emission probabilities we can say that State 1 is Consonant and State 0 is Vowel")

print("\nTask - 4")
model_nat = MultinomialHMM(n_components=2)
model_nat.transmat_ = trans_prob
model_nat.emissionprob_ = np.transpose(emmis_prob)
model_nat.startprob_ = np.array([0, 1])
scr2 = (model_nat.score(Data_arr))
scr1 = (model.score(Data_arr))
print("Log_Prob of Trained one is = " , scr1)
print("Log_Prob of Natural one is = " , scr2)
if(scr1 > scr2):
    print("Trained Model is better")
else:
    print("Natural Model is better")
print("Intializing the params of model from natural model")
model2 = MultinomialHMM(n_components=2,n_iter = 200)
model2.transmat_ = trans_prob
model2.emissionprob_ = np.transpose(emmis_prob)
model2.startprob_ = np.array([0, 1])
print("Training started")
model2.fit(Data_arr)
print("Training Done")
print("Model = ",model2.monitor_)
print("So the score has increased compared to th natural untrained one")
print("So if we compare the number of steps taken by the intialized and non-intialized one then intialzed took less steps to converge")
print("This is evident from printing of model.monitor_")

print("\nTask - 5")
#evaluate the test data
test_data = open("hmm-test.txt", "r").read()
test_data = preprocess(test_data)
test_data2 = convert(test_data)
Da = np.array(test_data2)
test_Data_arr = Da.reshape((Da.shape[0],1))
print("Log_Prob of test data on trained model",model2.score(test_Data_arr))
print("Log_Prob of test data on natural model",model_nat.score(test_Data_arr))
print("Probs are quite close to each other")
print("This suggest that trained model has overfitting")

