This is the repository of Voiceless Voice team for the AIC Competition 2

- Speech Preprocessing 
- Text Preprocessing
- Working Techniques 
- Feature Extraction 
- Model Building and training


------------------------------------------------------
Audio Preprocessing

- Remove noise from speech
- Trim audio to remove silence


-------------------------------------------------------
Remove non-Arabic Alphabet 

chars_to_ignore = ['>', ']', 'ل', 'ـ', '<', 'i', 'f', 'o', 'e', 'g', 'h', 'p', '=', '[', 'r', 'n', 'v', 'u', '÷'] # '١', '⁇', 'l', 'a'

After removing the symbols and English letters, the number of distinctive letters was reduced from 63 to 38


--------------------------------------------------------
Extract the Unique Alphabets

['إ', 'ش', 'ك', 'ح', 'ت', 'ف', 'خ', 'د', 'ر', 'ؤ', 'ء', 'ج', ' ', 'ً', 'ن', 'ع', 'م', 'ص', 'ث', 'ذ', 'غ', 'ه', 'ط', 'ى', 'ة', 'ض', 'ق', 'آ', 'ي', 'ز', 'ب', 'l', 'أ', 'س', 'و', 'ا', 'ئ', 'ظ']

38 unique alphabet


---------------------------------------------------------
Prepare the data for training

- Mel-frequency cepstral coefficients (MFCC)
- Chars tokenization (map every alphabet to a number)

{0: '<sos>', 1: 'ر', 2: 'l', 3: 'ً', 4: 'ا', 5: 'ب', 6: 'م', 7: 'ك', 8: 'ق', 9: 'ز', 10: 'ص', 11: 'آ', 12: 'ئ', 13: 'غ', 14: 'ى', 15: 'ط', 16: 'ؤ', 17: 'ت', 18: 'ظ', 19: 'ج', 20: 'ذ', 21: 'ش', 22: 'إ', 23: 'خ', 24: 'ن', 25: 'ة', 26: ' ', 27: 'ي', 28: 'ه', 29: 'ء', 30: 'ح', 31: 'و', 32: 'ث', 33: 'ع', 34: 'أ', 35: 'س', 36: 'ف', 37: 'د', 38: 'ض', 39: '<eos>'}


----------------------------------------------------------
Speech-Transformer                

Encoder(
  (lstm): LSTM(13, 128, batch_first=True, bidirectional=True)
  (pBLSTMs): Sequential(
    (0): pBLSTM(
      (blstm1): LSTM(512, 128, batch_first=True, bidirectional=True)
      (lock): LockedDropout()
    )
    (1): pBLSTM(
      (blstm1): LSTM(512, 128, batch_first=True, bidirectional=True)
      (lock): LockedDropout()
    )
    (2): pBLSTM(
      (blstm1): LSTM(512, 128, batch_first=True, bidirectional=True)
      (lock): LockedDropout()
    )
  )
  (key_network): Linear(in_features=256, out_features=128, bias=True)
  (value_network): Linear(in_features=256, out_features=128, bias=True)
)

Bidirectional LSTM:
The Bi-LSTM layer processes the input sequence in both forward and backward directions simultaneously. During the forward pass

Teacher forcing:
Teacher forcing acts like "training wheels." If the model makes a bad prediction, it is put back in place with the true value. Finally, we can make predictions using mixed teacher forcing.

