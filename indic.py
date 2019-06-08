from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

def sentiment(model,ts,length,text):
  #print(text)
  trn_seq=ts.texts_to_sequences([text])
  #print(trn_seq)
  padd=pad_sequences(trn_seq,maxlen=length,padding='post')
  yhat=model.predict(padd,verbose=0)
  percent_pos=yhat.argmax(axis=-1)
  percent_pos = yhat[0,0]
  if round(percent_pos) == 0:
    return 'NEGATIVE',percent_pos
  else:
    return 'POSITIVE',percent_pos


def asm_model():
  
  model = load_model('asm_char_model.h5')
  
  with open('asm_char_vector.pickle', 'rb') as handle:
    ts = pickle.load(handle)
    
    x=1000
  return model,ts,x


def hin_model():
  
  model = load_model('hindi_char_model.h5')
  
  with open('hindi_char_vector.pickle', 'rb') as handle:
    ts = pickle.load(handle)
    
    x=400
  return model,ts,x
