from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

app = Flask(__name__)

CORS(app)  




PATH = 'epoch=49-step=64150.ckpt'


#model class 
class Sequntial(pl.LightningModule):
    def __init__(self):
        super(Sequntial, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size = 3, hidden_size= 128,
                             batch_first = True, num_layers = 1)
        
        self.lstm2 = nn.LSTM(input_size = 128, hidden_size= 64,
                             batch_first = True, num_layers = 1)
        
        self.linear1 = nn.Linear(in_features=64 , out_features=25 )
        # self.relu = nn.ReLU()
        self.leakyrelu  = nn.LeakyReLU()
        self.linear2 = nn.Linear(in_features= 25, out_features=1)
        
    def forward(self,x):
       
        output, _ = self.lstm1(x)
        output, _ = self.lstm2(output)
        res = self.linear1(output)
        res = self.leakyrelu(res)
        # res = self.relu(res)
        return self.linear2(res)
            


# checkpoint = torch.load(PATH)
# model = torch.load(PATH)

model = Sequntial.load_from_checkpoint(checkpoint_path=PATH)
# model = Sequntial.load_from_checkpoint(checkpoint_path=path)  
            

@app.route('/predict', methods=['POST'])

def predict():
    try:
      
        data = request.get_json()
        print("Parsed Data:", data)
        print(data["age"])

        a= data['age']
        b=  data['rating']
        c = data['distance']
        
        age = float(a)  
        rating = float(b)
        distance = float(c)


       
        data =  torch.tensor([age,rating,distance], dtype=torch.float32)
        data = data.unsqueeze(0)
        print(data.shape)

        with torch.no_grad():
            logit = model(data)
            print(logit.item())
        
        return jsonify({'pred': round(logit.item(), 2)}), 200
        
    except Exception as e:
        print('Error:', str(e))
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5001 )