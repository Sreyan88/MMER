from torch import nn
from transformers import AutoConfig, AutoModel
import sys


class Transformer(nn.Module):
    def __init__(self, model, num_classes=4):
        super().__init__()

        self.name = model
        #config = AutoConfig.from_pretrained(self.name)
        #config.output_hidden_states = True
        #self.transformer = AutoModel.from_config(config)
        self.transformer = AutoModel.from_pretrained(self.name)
        self.nb_features = self.transformer.pooler.dense.out_features
        # self.pooler = nn.Sequential(
        #     nn.Linear(self.nb_features, self.nb_features), 
        #     nn.Tanh(),
        # )
        self.drop = nn.Dropout(p=0.1)
        self.logit = nn.Linear(self.nb_features, num_classes)

        

    def forward(self, input_ids = None, token_type_ids = None, attention_mask = None, labels = None):

        # B x Max_Length_Sentences = input_ids

        # B x Max_Length_Conversation (length of chain of comment, reply) x Max_Length_Sentences = input_ids --> Max_Length_Conversation (length of chain of comment, reply) x Max_Length_Sentences

        output = self.transformer(input_ids,attention_mask = attention_mask,return_dict=True)

        # Max_Length_Conversation (length of chain of comment, reply) x 768 for each loop

        # concat[output_transformer,output_graph] 768 x 2 (for each comment/reply) --> LSTM (hidden state of final comment) (B x Max_Length_Conversation x (768x2)) --> (B x lstm_hidden_size of last final layer)

        #print(output['pooler_output'].shape)
        #hidden_states = output['hidden_states']

        #print(hidden_states)
        
        #hidden_states = hidden_states[-1][:, 0] # Use the representation of the first token of the last layer
        ft = self.drop(output['pooler_output'])

        x = self.logit(ft)

        return x