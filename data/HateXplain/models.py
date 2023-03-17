import torch 
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
import torch.nn.functional as F

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class Model_Rational_Label(BertPreTrainedModel):
     def __init__(self,config):
        super().__init__(config)
        self.num_labels=2
        self.impact_factor=0.8
        self.bert = BertModel(config,add_pooling_layer=False)
        self.bert_pooler=BertPooler(config)
        self.token_dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()        
#         self.embeddings = AutoModelForTokenClassification.from_pretrained(params['model_path'], cache_dir=params['cache_path'])
        
     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, attn=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        # out = outputs.last_hidden_state
        out=outputs[0]
        logits = self.token_classifier(self.token_dropout(out))
        
        
#         mean_pooling = torch.mean(out, 1)
#         max_pooling, _ = torch.max(out, 1)
#         embed = torch.cat((mean_pooling, max_pooling), 1)
        embed=self.bert_pooler(outputs[0])
        y_pred = self.classifier(self.dropout(embed))
        loss_token = None
        loss_label = None
        loss_total = None
        
        if attn is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if mask is not None:
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, attn.view(-1), torch.tensor(loss_fct.ignore_index).type_as(attn)
                )
                loss_token = loss_fct(active_logits, active_labels)
            else:
                loss_token = loss_fct(logits.view(-1, 2), attn.view(-1))
            
            loss_total=self.impact_factor*loss_token
            
            
        if labels is not None:
            loss_funct = nn.CrossEntropyLoss()
            loss_logits =  loss_funct(y_pred.view(-1, self.num_labels), labels.view(-1))
            loss_label= loss_logits
            if(loss_total is not None):
                loss_total+=loss_label
            else:
                loss_total=loss_label
        if(loss_total is not None):
            return y_pred, logits, loss_total
        else:
            return y_pred, logits