import logging
import os

import torch
device_id = 0  # 假设要使用第二个CUDA设备
torch.cuda.device(device_id)
import numpy as np
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers import modeling_bert
    from transformers import modeling_roberta
except:
    from transformers.models.bert import modeling_bert
    from transformers.models.roberta import modeling_roberta
    from transformers.models.t5 import modeling_t5
from transformers import PretrainedConfig,AutoConfig,RobertaModel,T5Model,T5ForConditionalGeneration,T5Tokenizer
from mechanism import *
from transformers.file_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)
 
from modeling import modeling_gnn
from utils import layers
from utils import utils

logger = logging.getLogger(__name__)


# if os.environ.get('INHERIT_BERT', 0):
#     ModelClass = modeling_bert.BertModel
# else:
#     ModelClass = modeling_roberta.RobertaModel
ModelClass = modeling_t5.T5ForSequenceClassification
ModelClass_1=modeling_t5.T5EncoderModel
print ('ModelClass', ModelClass)


class SDPGNP(nn.Module):

    def __init__(self, args={}, model_name="roberta-large", k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=False, layer_id=-1,ep_1=1,ep_2=1):
        super().__init__()
        self.lmgnn = LMGNN(args, model_name, ep_1,ep_2,k, n_ntype, n_etype,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range, ie_dim=ie_dim, info_exchange=info_exchange, ie_layer_num=ie_layer_num,  sep_ie_layers=sep_ie_layers, layer_id=layer_id)

    # def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
    def batch_graph(self, edge_index_init, edge_type_init, pos_triples_init, neg_nodes_init, n_nodes):
        """
        edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        pos_triples = [[], [], []]
        for _i_ in range(n_examples):
            h = pos_triples_init[_i_][0] + _i_ * n_nodes #tensor[n_triple?,]
            r = pos_triples_init[_i_][1]                 #tensor[n_triple?,]
            t = pos_triples_init[_i_][2] + _i_ * n_nodes #tensor[n_triple?,]
            pos_triples[0].append(h)
            pos_triples[1].append(r)
            pos_triples[2].append(t)
        pos_triples = torch.stack([torch.cat(item) for item in pos_triples]) #[3, `total_n_triple`] where `total_n_triple` is sum of n_triple within batch
        assert pos_triples.size(0) == 3
        del h,r,t
        neg_nodes = [neg_nodes_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        neg_nodes = torch.cat(neg_nodes) #[`total_n_triple`, n_neg]
        assert neg_nodes.dim() == 2
        assert pos_triples.size(1) == neg_nodes.size(0)
        return edge_index, edge_type, pos_triples, neg_nodes
        # return edge_index, edge_type

    def forward(self, *inputs, epoch,cache_output=False, detail=False):
        """
        inputs_ids: (batch_size, num_choice, seq_len)    -> (batch_size * num_choice, seq_len)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        node_scores: [bs, nc, n_node, 1]
        adj_lengths: means the "actual" number of nodes (excluding padding)(batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )

        returns:
        logits: [bs, nc]
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        # _inputs = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:4]] + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[4:-2]] + [sum(x,[]) for x in inputs[-2:]]
        # *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type = _inputs
        _inputs = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:4]] + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[4:9]] + [sum(x,[]) for x in inputs[9:13]]
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type ,pos_triples, neg_nodes= _inputs
        node_scores = torch.zeros_like(node_scores)
        # edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        edge_index, edge_type ,pos_triples, neg_nodes= self.batch_graph(edge_index, edge_type ,pos_triples, neg_nodes,concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]
        lp_data = (pos_triples.to(node_type_ids.device), neg_nodes.to(node_type_ids.device))
        logits, link_loss = self.lmgnn(lm_inputs, concept_ids,
                                    node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,lp_data,epoch,
                                    emb_data=None, cache_output=cache_output)
        # logits: [bs * nc]
        link_losses = [item * bs for item in link_loss]
        logits = logits.view(bs, nc)
        if not detail:
            return logits, link_losses
        else:
            return logits,  concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig

    def check_outputs(self, logits, attn):
        bs = 4
        nc = 5
        assert logits.size() == (bs, nc)
        n_edges = 3


class LMGNN(nn.Module):

    def __init__(self, args={}, model_name="roberta-large", ep_1=1,ep_2=1,k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=False, layer_id=-1):
        super().__init__()
        config, _ = ModelClass.config_class.from_pretrained(
            model_name,
            cache_dir=None, return_unused_kwargs=True,
            force_download=False,
            output_hidden_states=True
        )
        self.init_range = init_range
        self.ep_1=ep_1
        self.ep_2=ep_2
        self.k = k
        self.concept_dim = concept_dim
        self.n_attention_head = n_attention_head
        if k >= 0:
            self.concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
            self.pooler = layers.MultiheadAttPoolLayer(n_attention_head, config.hidden_size, concept_dim)

        self.dropout_e = nn.Dropout(p_emb)

        if init_range > 0:
            self.apply(self._init_weights)

        self.mp, self.loading_info = TextKGMessagePassing.from_pretrained(model_name, output_hidden_states=True, output_loading_info=True, args=args, ep_1=self.ep_1,ep_2=self.ep_2,k=k, n_ntype=n_ntype, n_etype=n_etype, dropout=p_gnn, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers)

        self.layer_id = layer_id
        self.cpnet_vocab_size = n_concept
        self.LLM =ModelClass(config)
        for param in self.LLM.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        self.cp_emb=pretrained_concept_emb
        #FFN
        self.w_1 = nn.Linear(1024, config.d_model)#剖析点1
        self.w_2 = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout_1 = nn.Dropout(0.1)

        #link
        # self.linkpred = modeling_gnn.DistMultDecoder(args, num_rels=n_etype, h_dim=concept_dim)
        self.emb_LayerNorm=nn.LayerNorm(concept_dim)

        self.graph=[]


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,lp_data,epoch, emb_data=None, cache_output=False):
        """
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns:
        logits: [bs]
        """
        #LM inputs
        # memory_allocated_1 = torch.cuda.memory_allocated()
        # print("1.当前设备显存占用:", memory_allocated_1, "字节")
        input_ids, attention_mask, token_type_ids, output_mask = inputs

        # GNN inputs
        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2
        concept_ids=concept_ids.to('cuda:1')
        emb_id = torch.randint(0, self.cpnet_vocab_size, (concept_ids.shape[0], concept_ids.shape[1]))
        gnn_input = self.cp_emb[emb_id]
        # L_node = torch.max(torch.norm(gnn_input, dim=1)).item()
        # gnn_input=agg_dp(gnn_input,0.01,L_node)
        
        # gnn_input = gnn_input.to(self.LLM.device)
        # gnn_input = self.concept_emb(concept_ids - 1, emb_data)
        # gnn_input[:, 0] = 0
        # gnn_input = self.dropout_e(gnn_input) #(batch_size, n_node, dim_node)

        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

        # Merged core
        embedding_output,gnn_output= self.mp(epoch,input_ids, token_type_ids, attention_mask, output_mask, gnn_input, adj, node_type_ids, node_scores, special_nodes_mask, output_hidden_states=True)
        embedding_output=embedding_output.to(self.LLM.device)
        del gnn_input,emb_id
        gnn_output=gnn_output.to('cuda:1')
        #link_pred
        # pos_triples, neg_nodes = lp_data #pos_triples: [3, `total_n_triple`],  neg_nodes: [`total_n_triple`, n_neg]
        # pos_samples = pos_triples #[3, `total_n_triple`]
        # _n_neg = neg_nodes.size(1)
        # head_negative_sample = neg_nodes[:, :_n_neg//2]             #[`total_n_triple`, n_neg//2]
        # tail_negative_sample = neg_nodes[:, _n_neg//2:_n_neg//2*2]  #[`total_n_triple`, n_neg//2]
        # _bs, _, gnn_dim = gnn_output.size()
        # embs = gnn_output.view(-1, gnn_dim) #[`total_n_nodes`, gnn_dim]
        # embs = self.emb_LayerNorm(embs)
        # positive_score  = self.linkpred(embs, pos_samples) #[`total_n_triple`, 1]
        # head_neg_scores = self.linkpred(embs, (pos_samples, head_negative_sample), mode='head-batch')
        # tail_neg_scores = self.linkpred(embs, (pos_samples, tail_negative_sample), mode='tail-batch')
        # negative_score = torch.cat([head_neg_scores, tail_neg_scores], dim=-1) #[`total_n_triple`, total_n_neg]
        # scores = (positive_score, negative_score)

        # link_loss, pos_link_loss, neg_link_loss = self.linkpred.loss(scores)
        link_loss=0
        pos_link_loss=0
        neg_link_loss=0

        #池化
        edge_index, edge_type = adj
        # sent_vecs = gnn_output[:,0]
        # if epoch==6:
        #     if len(self.graph)<=200:
        #         self.graph.append({
        #             'prompt': sent_vecs,
        #             'adj_lengths': adj_lengths
        #         })

        # graph_vecs, pool_attn = self.pooler(sent_vecs, gnn_output, mask)
        # print(graph_vecs.shape)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        # print(gnn_output.device)
        prompts =self.w_2(self.dropout(F.relu(self.w_1(gnn_output))))
        prompts=prompts.to(self.LLM.device) 
        combined_input = torch.cat((prompts,embedding_output), dim=1)
        combined_input=combined_input.float()
        prefix_attention_mask = torch.ones(prompts.shape[0], prompts.shape[1]).to(node_scores.device)
        attention_mask = torch.cat((attention_mask,prefix_attention_mask), dim=1)
        attention_mask=attention_mask.to(self.LLM.device)
        deco=torch.full((prompts.shape[0],prompts.shape[1],prompts.shape[2]), 1).float()
        deco=deco.to(self.LLM.device)
        print(self.LLM.device)
        if epoch==20:
            if len(self.graph)<=200:
                self.graph.append({
                    'prompt': prompts,
                    'edge_index': edge_index
                })
        del embedding_output,prefix_attention_mask,prompts,edge_index,edge_type,adj,gnn_output
        outputs = self.LLM(
            attention_mask=attention_mask,
            inputs_embeds=combined_input,
            decoder_inputs_embeds=deco,
            return_dict=True,
        )
        

        del attention_mask,deco,combined_input
        
        # print(adj_lengths)
        pooled_output = outputs.logits
        pooled_output=pooled_output.to(node_scores.device)       
        pooled_output = self.dropout_1(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1,4)
        del pooled_output,logits,outputs
        return reshaped_logits, (link_loss, pos_link_loss, neg_link_loss)


    def check_outputs(self, logits, pool_attn):
        bs = 20
        assert logits.size() == (bs, 1)
        n_edges = 3


class TextKGMessagePassing(ModelClass_1):

    def __init__(self, config, args={}, ep_1=1,ep_2=1,k=5, n_ntype=4, n_etype=38, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        super().__init__(config=config)
        self.ep_1=ep_1
        self.ep_2=ep_2
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.hidden_size = concept_dim
        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim // 2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k

        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)
        self.activation = layers.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder_gnn = RoBERTaGAT(ep_1=self.ep_1,ep_2=self.ep_2,k=k, n_ntype=n_ntype, n_etype=n_etype, hidden_size=concept_dim, dropout=dropout, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers,h_s=config.hidden_size,num_hidden_layers=config.num_hidden_layers)

        self.sent_dim = config.hidden_size


    def forward(self, epoch,input_ids, token_type_ids, attention_mask, special_tokens_mask, H, A, node_type, node_score, special_nodes_mask, cache_output=False, position_ids=None, head_mask=None, output_hidden_states=True):
        """
        input_ids: [bs, seq_len]
        token_type_ids: [bs, seq_len]
        attention_mask: [bs, seq_len]
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
            edge_index: [2, n_edges]
            edge_type: [n_edges]
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        # LM inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 1D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        # if len(attention_mask.size()) == 2:
        #     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # elif len(attention_mask.size()) == 3:
        #     extended_attention_mask = attention_mask.unsqueeze(1)
        # else:
        #     raise ValueError("Attnetion mask should be either 1D or 2D.")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # if head_mask is not None:
        #     if head_mask.dim() == 1:
        #         head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #         head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
        #     elif head_mask.dim() == 2:
        #         head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        #     head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        # else:
        #     head_mask = [None] * self.config.num_hidden_layers

        # embedding_output = self.forward(input_ids, position_ids=position_ids, token_type_ids=token_type_ids).last_hidden_state

        #T5
        embedding = self.encoder(input_ids=input_ids, attention_mask=attention_mask,return_dict=True)
        embedding_output= embedding.last_hidden_state
        # GNN inputs
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = modeling_gnn.make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]
        _X=_X.to('cuda:2')
        embedding_output=embedding_output.to('cuda:2')
        edge_type=edge_type.to('cuda:2')
        _node_type=_node_type.to('cuda:2')
        _node_feature_extra=_node_feature_extra.to('cuda:2')
        # Merged core

        _X = self.encoder_gnn(epoch,embedding_output, _X, edge_index, edge_type, _node_type, _node_feature_extra)

        # GNN outputs
        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        return embedding_output,X

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
              - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
              - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
              - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                    - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)

        k = kwargs["k"]

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        print('here')
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    print('1')
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    print('2')
                    print(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME))
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    print('3')
                    # Load from a PyTorch checkpoint
                    print(WEIGHTS_NAME)
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                print('4')
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                print('5')
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                print('6')
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                print(resolved_archive_file)
                state_dict = torch.load(resolved_archive_file, map_location="cuda")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            all_keys = list(state_dict.keys())

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]

                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                    f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                    f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
                    f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                    f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                    f"and are newly initialized: {missing_keys}\n"
                    f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
                )
            else:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                    f"If your task is similar to the task the model of the ckeckpoint was trained on, "
                    f"you can already use {model.__class__.__name__} for predictions without further training."
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
                "all_keys": all_keys,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device:
            import torch_xla.core.xla_model as xm
            print( xm.xla_device())

            model = xm.send_cpu_data_to_device(model, 'cuda:1')
            model.to('cuda:1')

        return model



    def check_outputs(self, outputs, gnn_output):
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert gnn_output.size() == (bs, n_node, self.hidden_size)



class RoBERTaGAT(nn.Module):

    def __init__(self, ep_1=1,ep_2=1,k=5, n_ntype=4, n_etype=38, hidden_size=200, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False,h_s=1024,num_hidden_layers=1):
        super().__init__()
        self.ep_1=ep_1
        self.ep_2=ep_2
        self.k = k
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
        self.gnn_layers = nn.ModuleList([modeling_gnn.GATConvE(hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])
        self.activation = layers.GELU()
        self.dropout_rate = dropout

        self.sent_dim = h_s
        self.sep_ie_layers = sep_ie_layers
       
        self.concept_dim = concept_dim
        self.num_hidden_layers = num_hidden_layers
        self.info_exchange = info_exchange

        self.scale = hidden_size ** -0.5
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.proj_out = nn.Linear(h_s, hidden_size)
        self.hidden_size=hidden_size
        # self.graphs=[]

    def forward(self,epoch, hidden_states,  _X, edge_index, edge_type, _node_type, _node_feature_extra):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        head_mask: list of shape [num_hidden_layers]

        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        bs = hidden_states.size(0)
        # all_hidden_states = ()
        all_attentions = ()
        # print(self.num_hidden_layers)#24
        # print(self.k)#3
        for i in range(24):
            
            # print(i)
            # print(self.num_hidden_layers - self.k)
            # print(i)
            # LM
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            if i >= self.num_hidden_layers - 5:
                # print(i)
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                # _X = F.normalize(_X, p=2, dim=-1)
                _X,sensitivity = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra,self.ep_1,self.ep_2)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)
                # if epoch==6:
                #     if i==22:
                #         if len(self.graphs)<=200:
                #             self.graphs.append({
                #                 'prompt': _X,
                #                 'edge_index': edge_index
                #             })
                if i<=23:
                #GNP跨模态方法
                    
                    _X=_X.view(bs, -1, _X.size(1))
                    context = hidden_states[:, 0, :]
                    context=self.proj_out(context)
                    context=context.view(bs, -1, context.size(1))

                    # #graph做查询，context做键和值
                    Q = self.Wq(_X)
                    Q = F.normalize(Q, p=2, dim=-1)
                    K = self.Wk(context)
                    V = self.Wv(context)
                    K_norm = torch.max(torch.norm(K, dim=(1, 2)))
                    V_norm =torch.max(torch.norm(V, dim=(1, 2))) #torch.norm(V, p=2, dim=-1).max()
                    K_t = K.transpose(-2, -1)  # (batch_size, hidden_size, seq_len)
                    # 计算注意力分数 (Attention Scores)
                    att_weights = torch.matmul(Q, K_t) / (self.hidden_size ** 0.5)
                    print(att_weights.shape)
                    # sensitivity =float(K_norm*V_norm / (self.hidden_size ** 0.5)) 
                    ep_1=float(self.ep_1/4)
                    tmp=float(sensitivity /ep_1)
                    att=att_weights.cpu()
                    noise =np.random.laplace(0, tmp,att.size())
                    noise=torch.tensor(noise)
                    noise=noise.to(V.device)
                    att_weights=att_weights+noise
                    # att_weights = F.softmax(att_weights, dim=-1)
                    print(att_weights.shape)
                    out = torch.matmul(att_weights.double(), V.double())
                    _X=out.reshape(-1,_X.shape[2])
                    _X=_X.to(torch.float32)
                    del out,att_weights,Q,K,V,context
        return  _X 


    def check_outputs(self, outputs, _X):
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert _X.size() == (bs * n_node, self.concept_dim)



if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    utils.print_cuda_info()
    free_gpus = utils.select_free_gpus()
    device = torch.device("cuda:{}".format(free_gpus[0]))