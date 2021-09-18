# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP
from fairseq.modules import LayerNorm, PositionalEmbedding, FairseqDropout
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import RobertaHubInterface
from typing import Optional, List
from torch import Tensor
import copy
import math

logger = logging.getLogger(__name__)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CrossAttentionLayer(nn.Module):
    
    def __init__(self, d_model, kdim, vdim, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, 
                 residual=True, layernorm_embedding=False,
                 **kwargs
                ):
        super().__init__()
        
        self.residual = residual
        
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
      
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.normalize_before = normalize_before
    
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)     
        
        if layernorm_embedding:
            self.query_norm = nn.LayerNorm(d_model)
            self.key_norm = nn.LayerNorm(kdim)
        else:
            self.query_norm = nn.Sequential()
            self.key_norm = nn.Sequential()
            

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2 = self.multihead_attn(query=self.query_norm((self.with_pos_embed(tgt, query_pos))),
                                   key=self.key_norm(self.with_pos_embed(memory, pos)),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        if self.residual:
            tgt = tgt + self.dropout1(tgt2) 
        else:
            tgt = self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        raise NotImplementedError

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                **kwargs):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



@register_model("perceiver")
class PerceiverIOModel(FairseqEncoderModel):

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        # used for sentence classification, e.g.: GLUE benchmark
        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--number-of-latent", type=int, help="number of latent variables"
        )
        parser.add_argument(
            "--words-embed-dim",
            type=int,
            metavar="H",
            help="words embedding dimension",
        )
        parser.add_argument(
            "--lm-ffn-embed-dim",
            type=int,
            
            help="words embedding dimension for FFN",
        )
        parser.add_argument(
            "--latent-embed-dim",
            type=int,
            help="latent embedding dimension",
        )
        parser.add_argument(
            "--latent-ffn-embed-dim",
            type=int,
            metavar="F",
            help="latent embedding dimension for FFN",
        )
        parser.add_argument(
            "--latent-attention-heads",
            type=int,
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--lm-attention-heads",
            type=int,
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--mapping-residual",
            type=bool,
            help="residual connection in the mapping cross attention",
        )
        parser.add_argument(
            "--lm-attention-residual",
            type=bool,
            help="residual connection in the lm cross attention",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            "--min-params-to-wrap",
            type=int,
            metavar="D",
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = PerceiverEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )

        logger.info(x["args"])
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PerceiverEncoder(FairseqEncoder):
    """Perceiver encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        # set any missing default values
        base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))


        # Construct the latent array initial state.
        self.z_pos_enc = PositionalEmbedding(
                                        args.number_of_latent,
                                        args.latent_embed_dim,
                                        padding_idx=None,
                                        learned=True)
        
        # for word embedding
        self.embed_tokens = self.build_embedding(
            len(dictionary), args.words_embed_dim, dictionary.pad()
        )
        self.padding_idx = self.embed_tokens.padding_idx
        # if getattr(args, "layernorm_embedding", False):
        #     self.layernorm_embedding = LayerNorm(args.words_embed_dim)
        # else:
        #     self.layernorm_embedding = None
        
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.words_embed_dim,)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.words_embed_dim,
                self.embed_tokens.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        
        
        self.latent_mapping_layer = CrossAttentionLayer( args.latent_embed_dim, args.words_embed_dim, args.words_embed_dim,
                                                 args.latent_attention_heads, args.latent_ffn_embed_dim, args.dropout,
                                                 args.activation_fn, args.encoder_normalize_before, args.mapping_residual, 
                                                 getattr(args, "layernorm_embedding", False))
        
        encoder_norm = nn.LayerNorm(args.latent_embed_dim) 
        encoder_layer = TransformerEncoderLayer(args.latent_embed_dim, args.encoder_attention_heads, args.latent_ffn_embed_dim,
                                                args.dropout, args.activation_fn, args.encoder_normalize_before)
        self.processing_encoder = TransformerEncoder(encoder_layer, args.encoder_layers, encoder_norm)
        
        self.lm_decoder_layer = CrossAttentionLayer( args.words_embed_dim, args.latent_embed_dim, args.latent_embed_dim,
                                                 args.lm_attention_heads, args.lm_ffn_embed_dim, args.dropout,
                                                 args.activation_fn, args.encoder_normalize_before, args.lm_attention_residual,
                                                 getattr(args, "layernorm_embedding", False))
        
        self.latent_mapping_layer.apply(init_bert_params)
        self.processing_encoder.apply(init_bert_params)
        self.lm_decoder_layer.apply(init_bert_params)

        self.lm_head = self.build_lm_head(
            embed_dim=args.words_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra
    
    # def forward_embedding(
    #     self, src_tokens, token_embedding: Optional[torch.Tensor] = None):
    #     # embed tokens and positions
    #     if token_embedding is None:
    #         token_embedding = self.embed_tokens(src_tokens)
    #     x = embed = self.embed_scale * token_embedding
    #     if self.embed_positions is not None:
    #         x = embed + self.embed_positions(src_tokens)
    #     if self.layernorm_embedding is not None:
    #         x = self.layernorm_embedding(x)
    #     x = self.dropout_module(x)
    #     return x, embed

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        
        words_padding_mask = src_tokens.eq(self.padding_idx)
        # has_pads = (src_tokens.device.type == "xla" or words_padding_mask.any())
        
        words_embedding = self.embed_scale * self.embed_tokens(src_tokens).transpose(0, 1)
        words_pos = self.embed_positions(src_tokens).transpose(0, 1)
        # words_memory, words_embedding = self.forward_embedding(src_tokens, token_embeddings)
        
        latent_embedding = self.z_pos_enc.weight.unsqueeze(1).repeat(1, words_embedding.size(1), 1) # N, B, E
        tgt = torch.zeros_like(latent_embedding)
        
        # latent_variable = self.latent_mapping_layer(tgt, words_memory,
        #                                             memory_mask = words_padding_mask,
        #                                             pos=None,
        #                                             query_pos=latent_embedding,
        #                                             )
        latent_variable = self.latent_mapping_layer(tgt, words_embedding,
                                                    memory_key_padding_mask = words_padding_mask,
                                                    pos= words_pos,
                                                    query_pos=latent_embedding,
                                                    )
        
        process_out = self.processing_encoder(
            src=latent_variable,
            mask=None,src_key_padding_mask=None,
            pos=latent_embedding,
            
        )
        
        query_feature = torch.zeros_like(words_embedding)
        features = self.lm_decoder_layer( tgt = query_feature, 
                                       memory=process_out,
                                       pos=latent_embedding,
                                       query_pos=words_pos,
        )
        
        # T x B x C -> B x T x C
        features = features.transpose(0, 1)
        inner_states = None
        return features, {"inner_states": inner_states}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture("perceiver", "perceiver")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 26)
    args.number_of_latent = getattr(args, "number_of_latent", 256)
    args.words_embed_dim = getattr(args, "words_embed_dim", 768)
    args.latent_embed_dim = getattr(args, "latent_embed_dim", 1280)
    args.lm_ffn_embed_dim = getattr(args, "lm_ffn_embed_dim", 3072)
    args.latent_ffn_embed_dim = getattr(args, "latent_ffn_embed_dim", 1280)
    args.latent_attention_heads = getattr(args, "latent_attention_heads", 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.latent_attention_heads = getattr(args, "latent_attention_heads", 8)
    args.lm_attention_heads = getattr(args, "lm_attention_heads", 12)
    args.encoder_embed_dim = getattr(args, "words_embed_dim", 768) # for classifer

    args.mapping_residual = getattr(args, 'mapping_residual', True)
    args.lm_attention_residual = getattr(args, 'lm_attention_residual', True)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("perceiver", "perceiver_prenorm")
def perceiver_prenorm_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    base_architecture(args)


@register_model_architecture("perceiver", "perceiver_base")
def perceiver_base_architecture(args):
    base_architecture(args)


@register_model_architecture("perceiver", "perceiver_large")
def perceiver_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 40)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)

