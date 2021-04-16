from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import repeat

from src.models.components.featurizer import TokenFeaturizer


class SRCapsNet(nn.Module):
    def __init__(self,
                 *,
                 device,
                 n_slots,
                 n_intents,
                 d_dense: int,
                 d_sparse: int,
                 max_seq_len: int,
                 n_heads: int,
                 n_layers: int,
                 d_model: int,
                 d_semantic_space: int,
                 p_dropout_featurizer,
                 p_dropout_intent):
        super(SRCapsNet, self).__init__()

        # Set sizes and dimensions

        self.n_slots = n_slots
        self.n_intents = n_intents

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.d_semantic_space = d_semantic_space
        self.n_heads = n_heads

        self.device = device

        self.featurizer = TokenFeaturizer(d_dense=d_dense,
                                          d_sparse=d_sparse,
                                          d_model=d_model,
                                          p_dropout=p_dropout_featurizer)

        # Todo - This needs to be relative position encoding
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads) for _ in range(n_layers)])

        # w_activation_words (d, d) - Used if we want to learn the activation of words during training
        self.w_activations_words = nn.Parameter(torch.randn((self.d_model,
                                                             1)),
                                                requires_grad=True)

        # w_route_ws (max_seq_len, n_slot_caps, d_model) - Routing between word caps and slot caps
        self.w_route_ws = nn.Parameter(torch.randn((self.max_seq_len,
                                                    self.n_slots,
                                                    self.d_model)),
                                       requires_grad=True)

        # w_pose_ws (max_seq_len, n_slot_caps, d_model, d_model) - Pose between word caps and slot caps
        self.w_pose_ws = nn.Parameter(torch.randn((self.max_seq_len,
                                                   self.n_slots,
                                                   self.d_model,
                                                   self.d_model)),
                                      requires_grad=True)

        # w_route_si (n_slot_caps, n_intent_caps, d_model) - Routing between slot caps and intent caps
        self.w_route_si = nn.Parameter(torch.randn((self.n_slots,
                                                    self.n_intents,
                                                    self.d_model)),
                                       requires_grad=True)

        # w_pose_ws (n_slot_caps, n_intent_caps, d_model, d_model) - Pose between slot caps and intent caps
        self.w_pose_si = nn.Parameter(torch.randn((self.n_slots,
                                                   self.n_intents,
                                                   self.d_model,
                                                   self.d_model)),
                                      requires_grad=True)

        # Embedds
        # self.intent_embedding = nn.Linear(in_features=self.n_intents, out_features=self.d_semantic_space)
        # self.intent_embeddings(n_intents, d_semantic_space)
        self.all_intent_embeddings = nn.Parameter(torch.randn((self.n_intents,
                                                               self.d_semantic_space)),
                                                  requires_grad=True)

        self.cls_embedding = nn.Linear(in_features=2 * self.d_model, out_features=self.d_semantic_space)
        self.dropout_intent = nn.Dropout(p=p_dropout_intent)
        self.intent_norm = nn.LayerNorm(self.d_semantic_space)

    def forward(self, dense, sparse):
        """
        Takes the input of shape (b, max_seq_len) through the whole architecture.
        :return:
        """
        # tokens (b, max_seq_len, d_model)
        token_features = self.featurizer(dense, sparse)

        b, max_seq_len, _ = token_features.shape
        n_i = self.n_intents

        # cls_token(b, 1, d_model)
        # cls_token = torch.Tensor(torch.randn(b, 1, self.d_model))
        cls_token = torch.einsum('bld->bd', token_features).reshape((b, 1, -1))

        # token_features (b, max_seq_len + 1, d_model)
        token_features = torch.cat((token_features, cls_token), dim=-2)

        # encoded_embeddings (b, max_seq_len+1, d_model)
        for encoder_layer in self.encoder_layers:
            token_features = encoder_layer(token_features)

        # word_slots (b, max_seq_len, d_model)
        # cls_token  (b, 1, d_model) or (b, d_model)
        word_slots, cls_token = torch.split(token_features, self.max_seq_len, dim=1)

        # Get it to (b, d_model), if it's (b, 1, d_model)
        if cls_token.ndim == 3:
            cls_token = cls_token.squeeze(axis=1)

        c_ws, a_slots, u_slots = self.route_words_to_slots(token_features=word_slots)

        # slot_tags (b, max_seq_len, n_slots)
        slot_tags_predicted = c_ws

        # a_intents (b, n_intent_caps)
        # u_intents (b, n_intent_caps, d)
        a_intents, u_intents = self.route_slots_to_intents(a_slots, u_slots)

        # Todo - Figure how to do this with torch.gather
        max_activations = torch.argmax(a_intents, dim=-1, keepdim=False) + torch.arange(0, b * n_i, step=n_i).to(
            self.device)
        poses = u_intents.view(b * n_i, self.d_model).index_select(0, max_activations)

        # intents (b, d_model)
        intents_predicted = torch.cat((cls_token, poses), dim=-1)

        # Todo - Dropout?
        intents_predicted = self.intent_norm(self.cls_embedding(intents_predicted))

        # slots (b, max_seq_len, n_slots)
        # intent_prediction (b, d_semantic_space)
        return slot_tags_predicted, intents_predicted

    def route_words_to_slots(self,
                             token_features: torch.Tensor,
                             token_activation_strategy: str = 'weighted_sig'
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs self routing between the word capsules (post-encoder embeddings) and the slot capsules.
        :param token_activation_strategy: Experimental
               'norm' - Compute token activation as a mean over d_model
               'weighted' - Compute token activations with a learnable linear layer
        :param token_features: Encoded word embeddings for the current batch - (b, seq_len, d_model)
        :return: Pose vector and activation scalar, as follows: a_slots - (b, n_slots), u_slots - (b, n_slots, d_model)
        """

        # a_words (b, max_seq_len)
        if token_activation_strategy == 'norm':
            a_words = torch.einsum('bij -> bi', token_features) / self.d_model
        elif token_activation_strategy == 'weighted_sig':
            a_words = torch.einsum('bij, jk -> bik', token_features, self.w_activations_words)
            a_words = torch.sigmoid(torch.squeeze(a_words, dim=-1))
        elif token_activation_strategy == 'weighted':
            a_words = torch.einsum('bij, jk -> bik', token_features, self.w_activations_words)
            a_words = torch.squeeze(a_words, dim=-1)
        elif token_activation_strategy == 'ones':
            b, _ = token_features.shape
            a_words = torch.ones(())
        else:
            raise Exception('Invalid word activation calculation method')

        # w_route_ws (seq_len, n_slots, d)
        # token_features (b, seq_len, d)
        # c_ws_logits (b, seq_len, n_slot_caps)

        c_ws_logits = torch.einsum('lnj, blj -> bln', self.w_route_ws, token_features)

        # Softmax along the n_slots dimension
        # c_ws (b, seq_len, n_slot_caps)
        c_ws = f.softmax(c_ws_logits, dim=-1)

        # a_slots(b, n_slot_caps)
        a_numerator = torch.einsum('bls, bl -> bs', c_ws, a_words)
        sum_a_words = torch.einsum('bl -> b', a_words)

        a_denominator = repeat(sum_a_words, 'b -> b s', s=self.n_slots)
        a_slots = a_numerator / a_denominator

        # w_pose_ws (seq_len, n_slot_caps, d, d)
        # token_features (b, seq_len, d)
        # u_had_slots (b, seq_len, n_slot_caps, d)
        u_hat_slots = torch.einsum('lnij, blj -> blni', self.w_pose_ws, token_features)

        # u_slots (b, n_slots, d)
        u_numerator = torch.einsum('bls, bl, blsk -> bsk', c_ws, a_words, u_hat_slots)
        u_denominator = repeat(torch.einsum('bls, bl -> bs', c_ws, a_words), 'b i -> b i d', d=self.d_model)
        u_slots = u_numerator / u_denominator

        return c_ws, a_slots, u_slots

    def route_slots_to_intents(self,
                               a_slots: torch.Tensor,
                               u_slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs self routing between the slot capsules and the intent capsules.
        :param a_slots: Activation of slot capsules (b,
        :param u_slots:
        :return:
        """
        # w_route_si (n_slot_caps, n_intent_caps, d)
        # u_slots (b, n_slot_caps, d)
        # c_ws_logits (b, n_slot_caps, n_intent_caps)
        c_si_logits = torch.einsum('sij, bsj -> bsi', self.w_route_si, u_slots)

        # Softmax along the n_intent_caps dimension
        # c_si (b, n_slot_caps, n_intent_caps)
        c_si = f.softmax(c_si_logits, dim=-1)

        # Compute the weighted sum of the coupling coefficients
        # Multiply and sum along the "n_slots" dim -> weighed_c (b, n_intents)
        a_numerator = torch.einsum('bsi, bs -> bi', c_si, a_slots)

        # sum_a_slot shape (b, 1), repeat n_intent_caps times along axis 1
        sum_a_slots = torch.einsum('bs -> b', a_slots)

        a_denominator = repeat(sum_a_slots, 'b -> b i', i=self.n_intents) + 1e-8

        # a_intents (b, n_intent_caps)
        a_intents = a_numerator / a_denominator

        # w_pose_si (n_slot_caps, n_intent_caps, d, d)
        # u_slots (b, n_slots, d)
        u_hat_intents = torch.einsum('sijk, bsk -> bsik', self.w_pose_si, u_slots)

        # nominator: (b, n_intent_caps, d), denominator = (b, n_intent_caps, d)
        # u_intents: (b, n_intent_caps, d)
        u_numerator = torch.einsum('bsi, bs, bsik -> bik', c_si, a_slots, u_hat_intents)
        u_denominator = repeat(torch.einsum('bsi, bs -> bi', c_si, a_slots), 'b i -> b i d', d=self.d_model) + 1e-8
        u_intents = u_numerator / u_denominator

        return a_intents, u_intents

    def get_intent_semantic_embeddings(self):
        """
        Gets the tensor corresponding to the embedding of all intents in the low-dimensional semantic space.
        :return: The tensor associated with the nn.Parameter, of shape (n_intents, d_semantic)
        """
        return self.all_intent_embeddings.data
