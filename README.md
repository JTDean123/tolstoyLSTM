# tolstoyLSTM
Character generation with a stacked LSTM in Tensorflow trained on Anna Karenina.  

Default usage trains a model for 200 epochs with a batch size of 128, 500 nodes in each LSTM cell, and a sequence length of 50.  1000 character predictions are generated every 10 epochs, and the model can be trained by simply running (python3):  
```
python tolstoyLSTM.py
```

Training parameters can be changed via arguments. For example, a model can be trained for 50 epochs with 100 nodes per LSM cell with a batch size of 256 as follows:  
```
python tolstoyLSTM.py --epochs 50 --nodes 500 --batchsize 256
```

The generated text should start to sound (somewhat) coherent after ~50 epochs with default settings.  An example sentence generated after 200 epochs using the random seed:   
'''
ppiness, he walked with a slight swing on each leg
'''

The following output was generated:  
```
"what is it?" said stepan arkadyevitch, "youre weary for him."

"i dont understand. i expect his mothers are always
dull". 

"good-bye till at once, and short you though i made up of light how i could never the world", said levin, prince stepan arkadyevitch smiled.
```


