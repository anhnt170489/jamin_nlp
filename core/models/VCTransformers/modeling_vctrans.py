import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from core.models.VCTransformers.configuration_vctrans import VCTransConfig
from libs import PreTrainedModel


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


class VCTransPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = VCTransConfig
    base_model_prefix = "vctrans"

    def _init_weight(self, weight):
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)

    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)

    def _init_weights(self, m):
        """ Initialize the weights.
        """
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("AdaptiveEmbedding") != -1:
            if hasattr(m, "emb_projs"):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                self._init_weight(m.weight)
        elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
            if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
                self._init_weight(m.cluster_weight)
            if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
                self._init_bias(m.cluster_bias)
            if hasattr(m, "out_projs"):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, self.config.proj_init_std)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 1.0, self.config.init_std)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)


class VCTransEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = torch.nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # if position_ids is None:
        #     position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        #     position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
            self,
            config,
            r_r_bias=None,
            r_w_bias=None,
    ):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.n_head = config.num_attention_heads
        self.d_model = config.hidden_size
        self.d_head = config.hidden_size // config.num_attention_heads

        self.qkv_net = nn.Linear(self.d_model, 3 * self.n_head * self.d_head, bias=False)

        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.dropatt = nn.Dropout(config.attention_probs_dropout_prob)
        self.o_net = nn.Linear(self.n_head * self.d_head, self.d_model, bias=False)

        self.layer_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)

        self.scale = 1 / (self.d_head ** 0.5)

        self.pre_lnorm = config.pre_lnorm

        if r_r_bias is None or r_w_bias is None:  # Biases are not shared
            # self.r_r_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))
            # self.r_w_bias = nn.Parameter(torch.FloatTensor(self.n_head, self.d_head))

            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        else:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad_shape = (x.size(0), 1) + x.size()[2:]
        zero_pad = torch.zeros(zero_pad_shape, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded_shape = (x.size(1) + 1, x.size(0)) + x.size()[2:]
        x_padded = x_padded.view(*x_padded_shape)

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None, head_mask=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        # compute attention score
        # self.r_w_bias.data = self.r_w_bias.data.to(dtype=w_head_q.dtype)
        # self.r_r_bias.data = self.r_r_bias.data.to(dtype=w_head_q.dtype)

        rw_head_q = w_head_q + self.r_w_bias  # qlen x bsz x n_head x d_head
        rw_head_q = rw_head_q.to(dtype=w_head_q.dtype)
        AC = torch.einsum("ibnd,jbnd->ijbn", (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + self.r_r_bias
        rr_head_q = rr_head_q.to(dtype=w_head_q.dtype)
        BD = torch.einsum("ibnd,jnd->ijbn", (rr_head_q, r_head_k))  # qlen x klen x bsz x n_head

        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and torch.sum(attn_mask).item():
            attn_mask = attn_mask == 1  # Switch to bool
            if attn_mask.dim() == 2:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = (
                        attn_score.float().masked_fill(attn_mask[None, :, :, None], -65000).type_as(attn_score)
                    )
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[None, :, :, None], -1e30).type_as(attn_score)
            elif attn_mask.dim() == 3:
                if next(self.parameters()).dtype == torch.float16:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -65000).type_as(attn_score)
                else:
                    attn_score = attn_score.float().masked_fill(attn_mask[:, :, :, None], -1e30).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1).to(dtype=attn_score.dtype)
        attn_prob = self.dropatt(attn_prob)

        # Mask heads if we want to
        if head_mask is not None:
            attn_prob = attn_prob * head_mask

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        # if self.pre_lnorm:
        #     # residual connection
        #     outputs = [w + attn_out]
        # else:
        #     # residual connection + layer normalization
        outputs = self.layer_norm(w + attn_out)

        if self.output_attentions:
            outputs.append(attn_prob)
        outputs = (outputs, attn_prob) if self.output_attentions else (outputs,)

        return outputs


class VCTransSelfAttention(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.attn = RelPartialLearnableMultiHeadAttn(config, **kwargs)

    def forward(self, hidden_states, pos_emb, attention_mask=None, mems=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None):
        attn_outputs = self.attn(hidden_states, pos_emb, attn_mask=attention_mask, mems=mems, head_mask=head_mask)

        return attn_outputs


class VCTransSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VCTransAttention(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.self = VCTransSelfAttention(config, **kwargs)
        # self.output = VCTransSelfOutput(config)
        # self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
    #     heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
    #     for head in heads:
    #         # Compute how many pruned heads are before the head and move the index accordingly
    #         head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
    #         mask[head] = 0
    #     mask = mask.view(-1).contiguous().eq(1)
    #     index = torch.arange(len(mask))[mask].long()
    #
    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
    #
    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            pos_emb,
            attention_mask=None,
            mems=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, pos_emb, attention_mask=attention_mask, mems=mems, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask
        )
        # attention_output = self.output(self_outputs[0], hidden_states)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return self_outputs


class VCTransIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class VCTransOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VCTransLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.attention = VCTransAttention(config, **kwargs)
        # self.is_decoder = config.is_decoder
        # if self.is_decoder:
        #     self.crossattention = VCTransAttention(config, **kwargs)
        self.intermediate = VCTransIntermediate(config)
        self.output = VCTransOutput(config)

    def forward(
            self,
            hidden_states,
            pos_emb,
            attention_mask=None,
            mems=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, pos_emb, attention_mask=attention_mask, mems=mems,
                                                head_mask=head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # if self.is_decoder and encoder_hidden_states is not None:
        #     cross_attention_outputs = self.crossattention(
        #         attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        #     )
        #     attention_output = cross_attention_outputs[0]
        #     outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class VCTransLayerGroup(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layers = nn.ModuleList([VCTransLayer(config, **kwargs) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, pos_emb, mems=None, attention_mask=None, head_mask=None):
        # layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, layer in enumerate(self.layers):
            layer_output = layer(hidden_states, pos_emb, attention_mask=attention_mask, mems=mems,
                                 head_mask=head_mask[layer_index])
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            # if self.output_hidden_states:
            #     layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if self.output_hidden_states:
        #     outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        inv_freq = self.inv_freq.to(dtype=pos_seq.dtype)
        sinusoid_inp = torch.ger(pos_seq, inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class VCTransEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.pos_emb = PositionalEmbedding(self.config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        if not config.untie_r:
            n_head = config.num_attention_heads
            d_head = config.hidden_size // config.num_attention_heads
            self.r_w_bias = nn.Parameter(torch.FloatTensor(n_head, d_head))
            self.r_r_bias = nn.Parameter(torch.FloatTensor(n_head, d_head))
        self.layer_groups = nn.ModuleList([VCTransLayerGroup(config,
                                                             r_w_bias=None if config.untie_r else self.r_w_bias,
                                                             r_r_bias=None if config.untie_r else self.r_r_bias)
                                           for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, mems=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        qlen = hidden_states.shape[0]
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        # Preparing attention mask
        all_ones = hidden_states.new_ones((qlen, klen), dtype=torch.uint8)
        mask_len = klen - self.config.mem_len
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        if self.config.is_decoder:
            attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len))[:, :, None]  # -1
        else:
            all_ones[all_ones == 1] = 0
            all_ones[:, mask_len:] = 1
            attn_mask = all_ones[:, :, None]

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if self.config.clamp_len > 0:
            pos_seq.clamp_(max=self.config.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        all_attentions = ()

        # if self.output_hidden_states:
        all_hidden_states = []

        for i in range(self.config.num_hidden_layers):
            all_hidden_states.append(hidden_states)
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            mems_i = None if mems is None else mems[i]

            layer_group_output = self.layer_groups[group_idx](
                hidden_states,
                pos_emb,
                mems=mems_i,
                attention_mask=attn_mask,
                # head_mask=head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
                head_mask=head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group][0],
            )
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

        outputs = (hidden_states, all_hidden_states,)
        # if self.output_hidden_states:
        #     outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class VCTransPooler(nn.Module):
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


class VCTransModel(VCTransPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.mem_len = config.mem_len
        self.ext_len = config.ext_len

        self.embeddings = VCTransEmbeddings(config)
        self.encoder = VCTransEncoder(config)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = VCTransPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def init_mems(self, bsz):
        if self.config.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.config.num_hidden_layers):
                empty = torch.zeros(self.config.mem_len, bsz, self.config.hidden_size, dtype=param.dtype,
                                    device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mems=None,
            # position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
    ):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            bsz, qlen = input_ids.size()
        elif inputs_embeds is not None:
            bsz, qlen = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        encoder_hidden_states = None
        encoder_extended_attention_mask = None

        # # Prepare head mask if needed
        # # 1.0 in head_mask indicate we keep the head
        # # attention_probs has shape bsz x n_heads x N x N
        # # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # if head_mask is not None:
        #     if head_mask.dim() == 1:
        #         head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        #         head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
        #     elif head_mask.dim() == 2:
        #         head_mask = (
        #             head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        #         )  # We can specify head_mask for each layer
        #     head_mask = head_mask.to(
        #         dtype=next(self.parameters()).dtype
        #     )  # switch to fload if need + fp16 compatibility
        # else:
        # head_mask = [[None] * self.config.num_hidden_layers] * self.config.inner_group_num
        head_mask = [[None] * self.config.inner_group_num] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            # input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        if mems is None:
            mems = self.init_mems(bsz)

        encoder_outputs = self.encoder(
            embedding_output,
            mems,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        hids = encoder_outputs[1]
        mlen = mems[0].size(0) if mems is not None else 0
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        # We transpose back here to shape [bsz, len, hidden_dim]
        # outputs = [sequence_output.transpose(0, 1).contiguous(), new_mems]
        sequence_output = sequence_output.transpose(0, 1).contiguous()
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output, new_mems)
        if self.config.output_hidden_states:
            # Add last layer and transpose to library standard shape [bsz, len, hidden_dim]
            hids.append(sequence_output)
            hids = list(t.transpose(0, 1).contiguous() for t in hids)
            outputs += (hids,)
        if self.config.output_attentions:
            # Transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = encoder_outputs[-1]
            attentions = list(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            outputs += (attentions,)

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class VCTransPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(VCTransPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VCTransLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(VCTransLMPredictionHead, self).__init__()
        self.transform = VCTransPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class VCTransOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(VCTransOnlyMLMHead, self).__init__()
        self.predictions = VCTransLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class VCTransForMaskedLM(VCTransPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **masked_lm_loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **ltr_lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next token prediction loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """

    def __init__(self, config):
        super(VCTransForMaskedLM, self).__init__(config)

        self.vctrans = VCTransModel(config)
        self.cls = VCTransOnlyMLMHead(config)

        self.init_weights()

    def _adapt_extended_vocab(self):
        self.cls = VCTransOnlyMLMHead(self.config)
        self.cls.apply(self._init_weights)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                # position_ids=None,
                mems=None,
                head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, ):

        outputs = self.vctrans(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               # position_ids=position_ids,
                               mems=mems,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=encoder_attention_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


class VCTransForSequenceClassification(VCTransPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = VCTransForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.vctrans = VCTransModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            # position_ids=None,
            mems=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.vctrans(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            # position_ids=position_ids,
            mems=mems,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # pooled_output = outputs[1]
        pooled_output = torch.mean(outputs[0], 1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
