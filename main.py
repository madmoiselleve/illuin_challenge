from metrics import precision, balanced_prec
from models import *
from utils import *


## Load dataset + get a list of contexts and associated ids 
squad, documents = squad_json_to_dataframe('/Users/ezagury/Downloads/squad1.1/train-v1.1.json')

model = tf_idf_retriever()
model.fit(documents)

predictions = model.predict(squad['question'])

accuracy = precision(predictions, squad['c_id'])
balanced_acc = balanced_prec(predictions, squad['c_id'])

print(f'Accuracy of the model: {accuracy}')
print(f'Balanced accuracy of the model: {balanced_acc}')