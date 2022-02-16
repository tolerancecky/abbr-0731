# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules import TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .levenshtein_utils import (
    _apply_del_words,
    _apply_recover_words,
    _fill,
    _get_del_targets,
    _get_recover_label,
    _skip,
    _skip_encoder_out,
    random_abbr,
    full_abbr,
    _get_recover_label,
    _get_ins_targets,
    first_to_lastpad,
    get_real_src,
    get_real_encoderout,
)


@register_model("levenshtein_transformer")
class LevenshteinTransformerModel(FairseqNATModel):
    @property  #修饰方法，创建只读属性，后面不需接函数()执行
    def allow_length_beam(self):
        return False

    @staticmethod #返回函数的静态方法。从而可以实现实例化使用 C().f()，当然也可以不实例化调用该方法 C.f()。
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6",
            type=str,
            help="number of decoder layers before word_del, mask_ins, word_ins",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        # parser.add_argument(
        #     "--share-discriminator-maskpredictor",
        #     action="store_true",
        #     help="share the parameters for both mask-predictor and discriminator",
        # )
        parser.add_argument(
            "--sampling-for-deletion",
            action="store_true",
            help="instead of argmax, use sampling to predict the tokens",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = LevenshteinTransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False): #getattr(object, name[, default])返回一个对象属性值。
            decoder.apply(init_bert_params)
        return decoder
    #两者都是python中的可变参数。*args表示任何多个无名参数，它本质是一个tuple；**kwargs表示关键字参数，它本质上是一个dict；
    def forward(
        self, src_tokens, src_lengths,  prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # encoding
        # print(src_tokens)
        # print(src_lengths)
        # print(tgt_tokens)
        src_tokens=first_to_lastpad(src_tokens)
        l,encoderout_pad_mask,order,cut_off = get_real_src(src_tokens, 9, self.eos,self.pad)
        # tgt_lengths = tgt_tokens.ne(self.pad).sum(dim=1,dtype=torch.int32).reshape(-1,1).contiguous()
        # print(tgt_lengths)
        # print(src_tokens)
        # print(l)

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        encoder_out = get_real_encoderout(encoder_out,encoderout_pad_mask,order,cut_off)
        # print(encoder_out["encoder_out"][0].shape)

        # generate training labels for recover
        word_recover_target = _get_ins_targets(prev_output_tokens,tgt_tokens,self.pad)
        # print(prev_output_tokens)
        # print(word_recover_target)
        word_recover_masks = prev_output_tokens.ne(self.pad)
        word_recover_out, _ = self.decoder.forward_word_recover(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_recover_pred = word_recover_out.max(-1)[1].bool()
        # print(word_recover_pred)
        # print(l)
        word_predictions = prev_output_tokens.clone().masked_scatter_(word_recover_pred,l[word_recover_pred])
        # print(word_predictions)
        # print(src_tokens)
        # print(word_predictions)

        # generate training labels for abbr
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad) #掩码操作
        # print(word_del_targets)
        # print(tgt_tokens)
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            prev_output_tokens=word_predictions,
            encoder_out=encoder_out,
        )
        # print(word_del_out.max(-1)[1].bool())
        word_del_masks = word_predictions.ne(self.pad)

        return {
            "word_recover": {
                "out": word_recover_out,
                "tgt": word_recover_target,
                "mask": word_recover_masks,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_del": {
                "out": word_del_out,
                "tgt": word_del_targets,
                "mask": word_del_masks,
            },
                }


    def forward_decoder(
        self, decoder_out, encoder_out, src_tokens,max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history
        # print(src_tokens)
        src_tokens = first_to_lastpad(src_tokens)
        l, encoderout_pad_mask, order, cut_off = get_real_src(src_tokens, 9, self.eos, self.pad)
        # print(l)
        # print(encoder_out["encoder_out"][0].shape)
        # encoder_out = get_real_encoderout(encoder_out,encoderout_pad_mask,order,cut_off)

        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = torch.zeros_like(output_tokens).fill_(255) #fill_用255填充tensor
        else:
            if not encoder_out["encoder_padding_mask"]:
                max_src_len = encoder_out["encoder_out"].size(0)
                src_lens = encoder_out["encoder_out"].new(bsz).fill_(max_src_len)
            else:
                src_lens = (~encoder_out["encoder_padding_mask"][0]).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        # print(output_tokens)
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            word_del_score, word_del_attn = self.decoder.forward_word_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_del_word),
            )
            word_del_pred = word_del_score.max(-1)[1].bool()
            # print(word_del_pred)

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                word_del_attn,
                word_del_pred,
                self.pad,
                4,
                self.bos,
                self.eos,
            )
            # print(_tokens)
            # print(_scores)
            #进行删除操作再次更新
            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())


        # recover words
        # print(output_tokens)
        can_recover_word = output_tokens.eq(4).sum(1) > 0
        if can_recover_word.sum() != 0:
            word_recover_score, word_recover_attn = self.decoder.forward_word_recover(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_recover_word),
                encoder_out=_skip_encoder_out(self.encoder, encoder_out, can_recover_word),
            )
            word_recover_score, word_recover_pred = word_recover_score.max(-1)
            word_recover_pred = word_recover_pred.bool()
            # print(word_recover_pred)
            _tokens, _scores = _apply_recover_words(
                output_tokens[can_recover_word],
                output_scores[can_recover_word],
                word_recover_pred,
                l,
                word_recover_score,
                self.pad,
                4,
                self.bos,
                self.eos
            )
            # print(_tokens)
            # print(_scores)

            output_tokens = _fill(output_tokens, can_recover_word, _tokens, self.pad)
            # print(output_tokens)
            output_scores = _fill(output_scores, can_recover_word, _scores, 0)
            attn = _fill(attn, can_recover_word, word_recover_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        )

    #如果以tensor([[0, 2]])做初始化  #之后考虑用src_tokens做初始化
    def initialize_output_tokens(self, encoder_out, src_tokens):
        src_tokens = first_to_lastpad(src_tokens)
        l,encoderout_pad_mask,order,cut_off = get_real_src(src_tokens, 9, self.eos,self.pad)
        # encoder_out = get_real_encoderout(encoder_out,encoderout_pad_mask,order,cut_off)
        initial_output_tokens = full_abbr(l,self.pad,self.bos,self.eos,4)
        # initial_output_tokens,initial_output_scores = random_abbr(src_tokens, self.pad, self.bos, self.eos, 4)
        # initial_output_scores = initial_output_scores.type_as(encoder_out["encoder_out"][0])
        # initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        # #方便复制原来tensor的所有类型以及设备等等
        # initial_output_tokens[:, 0] = self.bos #起始符号
        # initial_output_tokens[:, 1] = self.eos  #结束符号
        # initial_output_tokens=src_tokens
        #
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

class LevenshteinTransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()#保留句子开头保留标识<bos>
        self.unk = dictionary.unk()#未知符号保留标识<unk>
        self.eos = dictionary.eos()#句尾保留标识<eos>
        #self.abbr = 4 #缩略词符号的id
        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        # self.embed_mask_ins = Embedding(255, self.output_embed_dim * 2, None) #表示K(MAX)取的是255
        self.embed_word_del = Embedding(2, self.output_embed_dim, None) #从0和1中抽出
        self.embed_word_recover = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]  #在此是6，6，6 控制decoder哪一层输出的特征
        #一般来说del 和 ins_mask可以在中间的block接分类器，但是ins_word这个任务比较有挑战性，因此一般取最后一个block
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        #nn.ModuleList加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中
        #layers_msk表示insert placeholder操作是由6个TransformerDecoderLayer组成的
        self.layers_recover = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_recover = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[1])
                ]
            )    #_表示占位符    表示通过6层 TransformerDecoderLayer
        #layers_del表示delete tokens操作是由6个TransformerDecoderLayer组成的
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[0])
                ]
            )  #_表示占位符    表示通过6层 TransformerDecoderLaye

        # if getattr(args, "share_discriminator_maskpredictor", False):
        #     assert getattr(
        #         args, "no_share_discriminator", False
        #     ), "must set saperate discriminator"
        #     self.layers_recover = self.layers_del
        #当"share_discriminator_maskpredictor"参数设置为FALSE时，表示两个共享，因此self.layers_mask和self.layers_del相等

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)previous decoder outputs of shape
                               `(batch, tgt_len)`, for teacher forcing
            encoder_out: a dictionary of hidden states and masks. output from the encoder,
                         used for encoder-side attention

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            #if self.embed_positions is not None
            
            #else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        
        # B x T x C -> T x B x C   B:batch size T:tgt_len C:
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None: #变成想要的emb_dim
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    # @ensemble_decoder
    # #指的是insert placeholders决定要每个空之间要插多少个[PLH]
    # def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
    #     features, extra = self.extract_features(
    #         prev_output_tokens,
    #         encoder_out=encoder_out,
    #         early_exit=self.early_exit[1],
    #         layers=self.layers_msk,
    #         **unused
    #     )
    #     features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
    #     decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
    #     if normalize:
    #         return F.log_softmax(decoder_out, -1), extra["attn"]
    #     return decoder_out, extra["attn"]

    # @ensemble_decoder
    # # 指的是fill-in tokens操作对每个[PLH]去预测应该插入词表中的哪个单词
    # def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
    #     features, extra = self.extract_features(
    #         prev_output_tokens,
    #         encoder_out=encoder_out,
    #         early_exit=self.early_exit[1],
    #         layers=self.layers,
    #         **unused
    #     )
    #     decoder_out = self.output_layer(features)  #output_layer 全连接层："Project features to the vocabulary size."
    #     if normalize:
    #         return F.log_softmax(decoder_out, -1), extra["attn"]
    #     return decoder_out, extra["attn"]

    @ensemble_decoder
    # 指的是delete tokens操作，对每个词进行二分类
    def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_del.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    # 指的是delete tokens操作，对每个词进行二分类
    def forward_word_recover(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_recover,
            **unused
        )
        decoder_out = F.linear(features, self.embed_word_recover.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

@register_model_architecture("levenshtein_transformer", "levenshtein_transformer")
def levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    # args.share_discriminator_maskpredictor = getattr(
    #     args, "share_discriminator_maskpredictor", False
    # )
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)


@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de"
)
#使用了上面的默认参数
def levenshtein_transformer_wmt_en_de(args):
    levenshtein_base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_vaswani_wmt_en_de_big"
)
def levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    levenshtein_base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture(
    "levenshtein_transformer", "levenshtein_transformer_wmt_en_de_big"
)
def levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    levenshtein_transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture(
    "levenshtein_transformer", "abbr_transformer_small"
)
def abbr_transformer_small(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.early_exit = getattr(args, "early_exit", "4,4,4")
    # args.no_share_discriminator = getattr(args, "no_share_discriminator", True)
    # args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", True)
    # args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", True)
    levenshtein_base_architecture(args)

@register_model_architecture(
    "levenshtein_transformer", "abbr_transformer_mid"
)
def abbr_transformer_mid(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 128)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 512)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.early_exit = getattr(args, "early_exit", "4,4,4")
    # args.no_share_discriminator = getattr(args, "no_share_discriminator", True)
    # args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", True)
    # args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", True)
    levenshtein_base_architecture(args)