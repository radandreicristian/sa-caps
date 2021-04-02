from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import repeat

from src.models.components.featurizer import TokenFeaturizer


class SRCapsNet(nn.Module):
    def __init__(self,
                 *,
                 pretrained_embeddings: dict,
                 sparse_features: dict,
                 n_slots,
                 n_intents,
                 max_seq_len: int = 30,
                 n_heads: int = 16,
                 n_layers: int = 2,
                 d_model: int = 512):
        super(SRCapsNet, self).__init__()

        # Set sizes and dimensions
        self.n_slots = n_slots
        self.n_intents = n_intents

        self.n_slot_caps = n_slots + 1  # total slots + "ctx" / "no-slot" capsule
        self.n_intent_caps = n_intents + 1  # TODO Do we want to model the unknown intent as well?

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Init various tensors and subnets

        self.word_featurizer = TokenFeaturizer(d_model=self.d_model)

        self.encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads) for _ in range(n_layers)])

        self.w_activations_words = nn.Parameter(torch.FloatTensor(self.d_model,
                                                                  self.d_model),
                                                requires_grad=True)

        self.w_route_ws = nn.Parameter(torch.FloatTensor(self.max_seq_len,
                                                         self.n_slot_caps,
                                                         self.d_model),
                                       requires_grad=True)

        self.w_pose_ws = nn.Parameter(torch.FloatTensor(self.max_seq_len,
                                                        self.n_slot_caps,
                                                        self.d_model,
                                                        self.d_model),
                                      requires_grad=True)

        self.w_route_si = nn.Parameter(torch.FloatTensor(self.n_slot_caps,
                                                         self.n_intent_caps,
                                                         self.d_model),
                                       requires_grad=True)

        self.w_pose_si = nn.Parameter(torch.FloatTensor(self.n_slot_caps,
                                                        self.n_intent_caps,
                                                        self.d_model,
                                                        self.d_model),
                                      requires_grad=True)

    def route_words_to_slots(self,
                             token_features: torch.FloatTensor,
                             token_activation_strategy: str = 'norm'
                             ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Route layer l (tokens) to layer l+1 (slots).
        :param token_activation_strategy: Experimental
               'norm' - Compute token activation as a mean over d_model
               'weighted' - Compute token activations with a learnable linear layer
        :param token_features: Encoded word embeddings for the current batch - (b, seq_len, d_model)
        :return: Pose vector and activation scalar, as follows: a_slots - (b, n_slots), u_slots - (b, n_slots, d_model)
        """

        # Todo - Cum putem sa compunem "activarile" la nivel de token? Putem tot in aceeasi maniera cumva...learnable
        if token_activation_strategy == 'norm':
            _, d = token_features.shape
            a_words = torch.einsum('bij -> bi', token_features) / d
        elif token_activation_strategy == 'weighted':
            a_words = torch.einsum('bij, jk -> bik', token_features, self.w_activations_words)
        else:
            raise Exception('Invalid word activation calculation method')

        # w_route_ws (seq_len, n_slot_caps, d)
        # token_features (b, seq_len, d)
        # c_ws_logits (b, seq_len, n_slot_caps)
        c_ws_logits = torch.einsum('lnj, blj -> bln', self.w_route_ws, token_features)

        # Softmax along the n_slot_caps dimension
        # c_ws (b, seq_len, n_slot_caps)
        c_ws = f.softmax(c_ws_logits, dim=-1)

        # a_slots(b, n_slot_caps)
        weighted_c = torch.einsum('bls, bl -> bs', c_ws, a_words)
        sum_a_words = torch.einsum('bl -> b', a_words)

        a_slots = weighted_c / repeat(sum_a_words, 'b -> bs', s=self.n_slot_caps)

        # w_pose_ws (seq_len, n_slot_caps, d, d)
        # token_features (b, seq_len, d)
        # u_had_slots (b, seq_len, n_slot_caps, d)
        u_hat_slots = torch.einsum('lnij, blj -> blni', self.w_pose_ws, token_features)

        # u_slots (b, n_slots, d)
        u_slots = torch.einsum('bls, bl, blsk -> bsk', c_ws, a_words, u_hat_slots) / repeat(
            torch.einsum('bls, bl -> bs', c_ws, a_words), 'bi -> bid', d=self.d_model)

        return a_slots, u_slots

    def route_slots_to_intents(self,
                               a_slots: torch.FloatTensor,
                               u_slots: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # w_route_si (n_slot_caps, n_intent_caps, d)
        # u_slots (b, n_slot_caps, d)
        # c_ws_logits (b, n_slot_caps, n_intent_caps)
        c_si_logits = torch.einsum('sij, bsj -> bsi', self.w_route_si, u_slots)

        # Softmax along the n_intent_caps dimension
        # c_si (b, n_slot_caps, n_intent_caps)
        c_si = f.softmax(c_si_logits, dim=-1)

        # Compute the weighted sum of the coupling coefficients
        # Multiply and sum along the "n_slots" dim -> weighed_c (b, n_intents)
        weighted_c = torch.einsum('bsi, bs -> bi', c_si, a_slots)

        # sum_a_slot shape (b, 1), repeat n_intent_caps times along axis 1
        sum_a_slots = torch.einsum('bs -> b', a_slots)

        # a_intents (b, n_intent_caps)
        a_intents = weighted_c / repeat(sum_a_slots, 'b -> bi', i=self.n_intent_caps)

        # w_pose_si (n_slot_caps, n_intent_caps, d, d)
        # u_slots (b, n_slots, d)
        u_hat_intents = torch.einsum('sijk, bsk -> bsik', self.w_pose_si, u_slots)

        # c_si (b, n_slot_caps, n_intent_caps)
        # a_slots (b, n_slot_caps)
        # u_hat_intents (b, n_slot_caps, n_intent_caps, d_model)
        # nominator: (b, n_intent_caps,
        u_intents = torch.einsum('bsi, bs, bsik -> bik', c_si, a_slots, u_hat_intents) / repeat(
            torch.einsum('bsi, bs -> bi', c_si, a_slots), 'bi -> bid', d=self.d_model)

        return a_intents, u_intents

    def forward(self, x):
        # x (b, max_batch_seq_len, chars_in_word)
        b, batch_max_seq_len, _ = x.shape
        assert (batch_max_seq_len <= self.max_seq_len, 'A sequence in the batch is too long.')

        # tokens(b,
        tokens = self.word_featurizer(x)

        pass
