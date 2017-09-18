To conclude my Deep Learning course, I decided to write a chatbot.I followed these steps.
1. Removed all special characters 
2. Converted all capital letters into small letters.
3. Added EOS tags
4. Converted  a disctionary and added all words and there respective integer ID's.
5. converted all sentences into  there respective integer sequences.
6. padded all the sentences to make all of them 21 in length.
7. feeded the sentences into a bidirectional encoder.
8. Used a uni directional decoder with initial state as the context vector from the encoder.
9. Fed previously generated tokens from the decoder as input to the decoder for the next step.
10 Used an Tensorflow helper in the chatbot_helper function instead  of the big looping function that I had written .
11. Working on attention mechanism, going to used a modified verison of the model_functions.py


Training took 4 hours, used interactive graph so as to produce output at the end to see how it worked.
Used the Cornell Movie dialouges corpus for training.

Please go through the code .


