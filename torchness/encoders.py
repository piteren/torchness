import torch
from typing import Optional, Tuple, Union

from torchness.base import ACT, INI, TNS, DTNS, TorchnessException
from torchness.initialize import bert_initializer, my_initializer
from torchness.layers import LayDense, TF_Dropout, LayConv1D, LayRES, zeroes


class LayBlockDRT(torch.nn.Module):
    """ Block (Layer) of EncDRT
    LN > Dense or two with drop in between > drop > RES with drop """

    def __init__(
            self,
            in_width: int,
            in_lay_norm=                True,           # input layer norm
            do_scaled_dns: bool=        False,          # two denses (True) or single dense (False)
            dns_scale: int=             4,              # first dense up-scale
            activation: ACT=            torch.nn.ReLU,
            interlay_dropout: float=    0.0,            # dropout in between two denses
            lay_dropout: float=         0.0,            # dropout after dense/s
            residual: bool=             True,           # residual yes/no
            res_dropout: float=         0.0,            # dropout on residual connection
            do_zeroes: bool=            True,
            initializer: INI=           None):

        super().__init__()

        self.do_zeroes = do_zeroes

        if initializer is None:
            initializer = my_initializer

        self.ln_in = torch.nn.LayerNorm(normalized_shape=in_width) if in_lay_norm else None

        self.denses = []
        self.drop_interlay = None
        if do_scaled_dns:
            # dense (scale up) with activation
            self.denses.append(LayDense(
                in_features=    in_width,
                out_features=   in_width * dns_scale,
                activation=     activation,
                bias=           True,
                initializer=    initializer))
            self.add_module(f'dense0', self.denses[-1])

            if interlay_dropout:
                self.drop_interlay = torch.nn.Dropout(p=interlay_dropout)

            # dense (scale down) without activation
            self.denses.append(LayDense(
                in_features=    in_width * dns_scale,
                out_features=   in_width,
                activation=     None,
                bias=           True,
                initializer=    initializer))
            self.add_module(f'dense1', self.denses[-1])
        else:
            # just single dense, with activation
            self.denses.append(LayDense(
                in_features=    in_width,
                out_features=   in_width,
                activation=     activation,
                bias=           True,
                initializer=    initializer))
            self.add_module(f'dense0', self.denses[-1])

        self.drop_lay = torch.nn.Dropout(p=lay_dropout) if lay_dropout else None

        self.res = LayRES(in_features=in_width, dropout=res_dropout) if residual else None

    def forward(self, inp:TNS) -> DTNS:

        zs = None

        out = inp
        if self.ln_in:
            out = self.ln_in(out)

        out = self.denses[0](out)
        if self.do_zeroes:
            zs = zeroes(out).detach()

        if len(self.denses) > 1: # there is second one, without activation

            if self.drop_interlay:
                out = self.drop_interlay(out)

            out = self.denses[1](out)

        if self.drop_lay:
            out = self.drop_lay(out)

        if self.res:
            out = self.res(inp=out, bypass=inp)

        return {
            'out':      out,
            'zeroes':   zs}


class EncDRT(torch.nn.Module):
    """ Deep Residual encoder based on stacked LayBlockDRT """

    def __init__(
            self,
            in_width: int,
            in_lay_norm: bool=          True,           # input LN
            in_dropout: float=          0.0,            # dropout on input
            n_layers: int=              6,              # number of blocks
            shared_lays: bool=          False,          # shared variables in enc_layers
            lay_width: Optional[int]=   None,           # for None matches input width
            do_scaled_dns: bool=        True,
            dns_scale: int=             4,
            activation: ACT=            torch.nn.ReLU,
            interlay_dropout=           0.0,
            lay_dropout: float=         0.0,
            residual: bool=             True,
            res_dropout: float=         0.0,
            lay_norm=                   True,           # LN in each LayBlockDRT (input)
            do_zeroes: bool=            True,
            initializer: INI=           None):

        super().__init__()

        self.do_zeroes = do_zeroes

        if initializer is None:
            initializer = my_initializer

        self.ln_in = torch.nn.LayerNorm(normalized_shape=in_width) if in_lay_norm else None

        self.in_drop_lay = torch.nn.Dropout(p=in_dropout) if in_dropout else None

        self.in_width = in_width
        self.lay_width = lay_width or self.in_width
        self.projection_lay = LayDense(
            in_features=    self.in_width,
            out_features=   self.lay_width,
            activation=     None,
            bias=           False,
            initializer=    initializer) if self.lay_width != self.in_width else None

        num_layers_to_build = 1 if shared_lays else n_layers
        self.drt_lays = [LayBlockDRT(
            in_width=           self.lay_width,
            in_lay_norm=        lay_norm,
            do_scaled_dns=      do_scaled_dns,
            dns_scale=          dns_scale,
            activation=         activation,
            interlay_dropout=   interlay_dropout,
            lay_dropout=        lay_dropout,
            residual=           residual,
            res_dropout=        res_dropout,
            do_zeroes=          self.do_zeroes,
            initializer=        initializer
        ) for _ in range(num_layers_to_build)]
        for lix,lay in enumerate(self.drt_lays):
            self.add_module(f'lay_drt_{lix}',lay)
        if shared_lays and n_layers > 1:
            self.drt_lays *= n_layers

    def forward(self, inp:TNS) -> DTNS:

        zsL = []

        out = inp

        if self.ln_in:
            out = self.ln_in(out)

        if self.in_drop_lay: # input dropout
            out = self.in_drop_lay(out)

        if self.projection_lay: # input projection, no activation <- do not catch zeroes
            out = self.projection_lay(out)

        for drt_lay in self.drt_lays:
            lay_out = drt_lay(out)
            out = lay_out['out']
            if self.do_zeroes:
                zsL.append(lay_out['zeroes'])

        return {
            'out':      out,
            'zeroes':   torch.cat(zsL).detach() if self.do_zeroes else None}


class LayBlockCNN(torch.nn.Module):
    """ Block (Layer) of EncCNN
    LN > CNN > act > drop > RES > LayBlockDRT

    in general input for LayBlockCNN has a shape of:
                [feats] - single element (token == sequence of lenht 1)
           [seq, feats] - sequence of seq tokens
    [batch, seq, feats] - batch of sequences (each of seq tokens)

    * be careful about batch of single token sequences, input should have shape:
        [batch, 1, feats]  (NOT: [batch, feats] <- this is a single sequence)

    history for casual encoder has in general shape:
    [batch, kernel_size-1, feats]

    zero_history (initial history) is a tensor of zeroes of shape:
    [kernel_size-1, feats], its batch size will be extended to the given input

    EncCNN - number of parameters: kernel*in_features*n_filters """

    def __init__(
            self,
            n_filters: int,                             # num of filters
            padded=                     True,           # if not padded reduces sequence length
            kernel_size: int=           3,              # layer kernel
            activation: ACT=            torch.nn.ReLU,  # global enc activation func
            lay_dropout: float=         0.0,
            res_dropout: float=         0.0,
            # lay_DRT
            do_ldrt=                    False,          # lay DRT - build or not
            ldrt_do_scaled_dns: bool=   True,
            ldrt_dns_scale: int=        4,
            ldrt_drop: float or None=   0.0,
            ldrt_residual: bool=        True,
            ldrt_res_dropout: float=    0.0,
            # other
            do_zeroes: bool=            True,
            detach_history: bool=       True,           # by default state (history) will be detached on output
            initializer: INI=           None):

        super().__init__()

        self.n_filters = n_filters
        self.padded = padded
        if kernel_size % 2 == 0: raise TorchnessException('LayBlockCNN kernel_size cannot be even number')
        self.kernel_size = kernel_size
        self.do_zeroes = do_zeroes

        self.lay_ln = torch.nn.LayerNorm(normalized_shape=self.n_filters)

        self.lay_conv1D = LayConv1D(
            in_features=    self.n_filters,
            n_filters=      self.n_filters,
            kernel_size=    self.kernel_size,
            padding=        'valid',
            activation=     None,
            initializer=    initializer)

        self.activation = activation() if activation else None

        self.lay_drop = torch.nn.Dropout(p=lay_dropout) if lay_dropout else None

        self.res = LayRES(
            in_features=    self.n_filters,
            dropout=        res_dropout)

        self.lay_DRT = LayBlockDRT(
            in_width=       self.n_filters,
            do_scaled_dns=  ldrt_do_scaled_dns,
            dns_scale=      ldrt_dns_scale,
            activation=     activation,
            lay_dropout=    ldrt_drop,
            residual=       ldrt_residual,
            res_dropout=    ldrt_res_dropout,
            do_zeroes=      self.do_zeroes,
            initializer=    initializer) if do_ldrt else None

        self.detach_history = detach_history

    def get_zero_history(self) -> TNS:
        """ prepares initial history for casual mode
        zero_history shape: [kernel_size-1, n_filters] """
        return torch.zeros([self.kernel_size-1, self.n_filters])

    def forward(
            self,
            inp: TNS,
            history: Optional[TNS]= None,  # history must be given for casual mode
    ) -> DTNS:

        inp_orig_shape = inp.shape

        # expand single token
        if len(inp_orig_shape) == 1:
            inp = inp.expand([1, inp_orig_shape[0]])

        inp_shape = inp.shape

        zsL = []
        out = self.lay_ln(inp)

        state = None

        # concatenate with history on the left -> casual block
        if history is not None:

            history_shape = history.shape

            if history_shape[-1] != inp_shape[-1] or history_shape[-2] != self.kernel_size-1:
                raise TorchnessException(f'wrong history shape, given: {history_shape}')

            if list(inp_shape[:-2]) != list(history_shape[:-2]):
                history = history.expand(list(inp_shape[:-2]) + list(history_shape))

            out = torch.concat([history.to(out.device, out.dtype), out], dim=-2)

            # prepare state
            state_spl = torch.split(
                tensor=                 out,
                split_size_or_sections= [out.size(-2)-self.kernel_size+1, self.kernel_size-1],
                dim=                    -2)
            state = state_spl[-1]

        # pad both sides
        if history is None and self.padded:

            pad_shape = list(inp_shape)
            pad_shape[-2] = (self.kernel_size-1) // 2
            pad = torch.zeros(pad_shape).to(out.device, out.dtype)

            out = torch.concat([pad, out, pad], dim=-2)

        out = self.lay_conv1D(out)

        if self.activation:
            out = self.activation(out)
            if self.do_zeroes:
                zsL.append(zeroes(out))

        if self.lay_drop:
            out = self.lay_drop(out)

        # it is not possible to do RES for not padded version
        if self.padded:
            out = self.res(inp=out, bypass=inp)

        if self.lay_DRT:
            lay_out = self.lay_DRT(out)
            out = lay_out['out']
            if self.do_zeroes:
                zsL.append(lay_out['zeroes'])

        if len(inp_orig_shape) == 1:
            out = torch.squeeze(out, dim=0)

        if state is not None and self.detach_history:
            state = state.detach()

        return {
            'out':      out,
            'state':    state,
            'zeroes':   torch.cat(zsL).detach() if self.do_zeroes else None}


class EncCNN(torch.nn.Module):
    """ CNN 1D Encoder (for sequences)
    encoder built of stacked layers of LayBlockCNN

    EncCNN - number of parameters: projection + n_layers*LayBlockCNN """

    def __init__(
            self,
            in_features: int,                           # input num of channels
            time_drop: float=           0.0,
            feat_drop: float=           0.0,
            # layer
            shared_lays: bool=          False,          # shared variables between LayBlockCNN
            n_layers :int=              6,              # num of layers
            padded=                     True,           # if not padded reduces sequence length
            kernel_size :int=           3,              # layer kernel
            n_filters :Optional[int]=   None,           # num of filters, for None uses in_features
            activation: ACT=            torch.nn.ReLU,  # global enc activation func
            lay_dropout: float=         0.0,
            res_dropout: float=         0.0,
            # lay_DRT
            do_ldrt=                    False,          # lay DRT - build or not
            ldrt_do_scaled_dns: bool=   True,
            ldrt_dns_scale: int=        4,
            ldrt_drop: float or None=   0.0,
            ldrt_residual: bool=        True,
            ldrt_res_dropout: float=    0.0,
            # other
            do_zeroes: bool=            True,
            detach_history: bool=       True,           # by default state (history) will be detached in output
            initializer: INI=           None):

        super(EncCNN, self).__init__()

        self.in_features = in_features
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.n_filters = n_filters or self.in_features
        self.do_zeroes = do_zeroes

        self.in_TFdrop_lay = TF_Dropout(
            time_drop=  time_drop,
            feat_drop=  feat_drop) if time_drop or feat_drop else None

        self.projection_lay = LayDense(
            in_features=    self.in_features,
            out_features=   self.n_filters,
            activation=     None,
            bias=           False,
            initializer=    initializer) if self.in_features != self.n_filters else None

        num_blocks_to_build = 1 if shared_lays else self.n_layers

        self.blocks = [LayBlockCNN(
            n_filters=          self.n_filters,
            padded=             padded,
            kernel_size=        self.kernel_size,
            activation=         activation,
            lay_dropout=        lay_dropout,
            res_dropout=        res_dropout,
            do_ldrt=            do_ldrt,
            ldrt_do_scaled_dns= ldrt_do_scaled_dns,
            ldrt_dns_scale=     ldrt_dns_scale,
            ldrt_drop=          ldrt_drop,
            ldrt_residual=      ldrt_residual,
            ldrt_res_dropout=   ldrt_res_dropout,
            do_zeroes=          self.do_zeroes,
            detach_history=     detach_history,
            initializer=        initializer) for _ in range(num_blocks_to_build)]

        for bix,block in enumerate(self.blocks): self.add_module(f'block_{bix}',block)

        if shared_lays and self.n_layers > 1: self.blocks *= self.n_layers

        self.out_ln = torch.nn.LayerNorm(normalized_shape=self.n_filters)

    def get_zero_history(self) -> TNS:
        """ prepares initial history for casual mode
        zero_history shape: [n_layers, kernel_size-1, n_filters] """
        block_zero_history = self.blocks[0].get_zero_history()
        return block_zero_history.expand([self.n_layers] + list(block_zero_history.shape))

    def forward(
            self,
            inp: TNS,
            history: Optional[TNS]= None, # history must be given for casual mode
    ) -> DTNS:

        inp_orig_shape = inp.shape

        # expand single token (it will disable expansion in every block -> speed-up)
        if len(inp_orig_shape) == 1:
            inp = inp.expand([1, inp_orig_shape[0]])

        states = []  # here we will store block states to concatenate them finally
        zsL = [] if self.do_zeroes else None

        if self.in_TFdrop_lay:
            inp = self.in_TFdrop_lay(inp)

        if self.projection_lay:
            inp = self.projection_lay(inp)

        output = inp

        histories = torch.split(history, 1, dim=-3) if history is not None else [None]*self.n_layers

        for block, hist in zip(self.blocks, histories):

            if hist is not None:
                hist = torch.squeeze(hist, dim=-3)

            block_out = block(output, history=hist)
            output = block_out['out']

            if block_out['state'] is not None:
                states.append(torch.unsqueeze(block_out['state'], dim=-3))

            if self.do_zeroes:
                zsL.append(block_out['zeroes'])

        output = self.out_ln(output)

        if len(inp_orig_shape) == 1:
            output = torch.squeeze(output, dim=0)

        if zsL:
            zsL = torch.cat(zsL).detach()

        return {
            'out':      output,
            'state':    torch.cat(states, dim=-3) if states else None,
            'zeroes':   zsL}


class MyMHA(torch.nn.MultiheadAttention):
    """ QKV_linear_projection + QKV_scaled_dot_product_attention + linear_out_projection """

    # replaces xavier with bert_initializer
    def _reset_parameters(self):

        if self._qkv_same_embed_dim:
            bert_initializer(self.in_proj_weight)
        else:
            bert_initializer(self.q_proj_weight)
            bert_initializer(self.k_proj_weight)
            bert_initializer(self.v_proj_weight)
        bert_initializer(self.out_proj.weight)

        if self.in_proj_bias is not None:
            torch.nn.init.zeros_(self.in_proj_bias)
            torch.nn.init.zeros_(self.out_proj.bias)
        if self.bias_k is not None:
            bert_initializer(self.bias_k)
        if self.bias_v is not None:
            bert_initializer(self.bias_v)


class LayBlockTNS(torch.nn.Module):
    """ Block (Layer) of EncTNS
    based on torch.nn.modules.transformer.TransformerEncoderLayer """

    def __init__(
            self,
            d_model: int=       512,
            nhead: int=         8,
            dns_scale: int=     4,              # up-scale for first dense of two
            dropout: float=     0.1,
            dropout_att: float= 0.0,            # in original (torch.nn..) implementation dropout_att == dropout
            activation: ACT=    torch.nn.ReLU,
            dropout_res: float= 0.0,            # dropout on residual bypass
            do_zeroes: bool=    True):

        super().__init__()

        self.norm1 = torch.nn.LayerNorm(normalized_shape=d_model)

        self.self_attn = MyMHA(
            embed_dim=      d_model,
            num_heads=      nhead,
            dropout=        dropout_att,
            bias=           True,
            add_bias_kv=    False,
            add_zero_attn=  False,
            kdim=           None,
            vdim=           None,
            batch_first=    True)

        self.dropout1 = torch.nn.Dropout(p=dropout) if dropout else None

        self.res1 = LayRES(in_features=d_model, dropout=dropout_res)

        # (LN > Dense or two_with_drop_in_between > drop > RES with drop)
        self.lay_drt = LayBlockDRT(
            in_width=           d_model,
            do_scaled_dns=      True,
            dns_scale=          dns_scale,
            activation=         activation,
            interlay_dropout=   dropout, # INFO: TF implementation has not this dropout
            lay_dropout=        dropout,
            residual=           True,
            res_dropout=        dropout_res,
            do_zeroes=          do_zeroes,
            initializer=        bert_initializer)

    def forward(
            self,
            inp: TNS,
            task_query: Optional[TNS]=              None,           # forces task-attention mode (TAT)
            inp_mask: Optional[TNS]=                None,
            inp_key_padding_mask: Optional[TNS]=    None) -> DTNS:

        x = inp

        if task_query is None: x = self.norm1(x) # norm first https://arxiv.org/pdf/2002.04745v1.pdf
        else: task_query = self.norm1(task_query) # LN on task_query

        x = self.self_attn(
            query=              x if task_query is None else task_query,
            key=                x,
            value=              x,
            key_padding_mask=   inp_key_padding_mask,
            need_weights=       False,
            attn_mask=          inp_mask)[0]

        if self.dropout1:
            x = self.dropout1(x)

        bypass = inp if task_query is None else task_query
        x = self.res1(inp=x, bypass=bypass)

        return self.lay_drt(x)


class EncTNS(torch.nn.Module):
    """ Transformer Encoder
    based on torch.nn.modules.transformer.TransformerEncoder
    + Task Attention Transformer (TAT)
    + pyramidal encoding """

    def __init__(
            self,
            num_layers: int=                        6,
            num_layers_TAT: int=                    0,
            initial_TAT_avg: bool=                  True,   # how to prepare first block TAT task_query
            shared_lays: Optional[Tuple[int,...]]=  None,   # tuple defines layers groups with shared variables, e.g.: (2,2,2)
            max_seq_len: Optional[int]=             None,   # when given (int) adds positional embeddings (PE) to seq
            # block params
            d_model: int=                           512,
            nhead: int=                             8,
            dns_scale: int=                         4,
            dropout: float=                         0.1,
            dropout_att: float=                     0.0,
            activation: ACT=                        torch.nn.ReLU,
            dropout_res: float=                     0.0,
            do_zeroes: bool=                        True,
    ):

        super().__init__()

        self.d_model = d_model
        self.do_zeroes = do_zeroes

        # positional embeddings (trainable)
        self.pos_emb = None
        if max_seq_len:
            self.pos_emb = torch.nn.Parameter(data=torch.empty(size=(max_seq_len, self.d_model)))
            bert_initializer(self.pos_emb)

        self.initial_task_query = None
        if not initial_TAT_avg:
            self.initial_task_query = torch.nn.Parameter(
                data=           torch.empty(self.d_model),
                requires_grad=  False) # TODO: check with experiments
            bert_initializer(self.initial_task_query)

        # manage layers number
        num_layers_to_build = num_layers + num_layers_TAT
        if shared_lays is not None:

            if not sum(shared_lays) == num_layers_to_build:
                raise TorchnessException('Sum of shared layers must be equal to total num of layers (num_layers + num_layers_TAT)')

            num_layers_to_build = len(shared_lays)

        layers = [LayBlockTNS(
            d_model=        self.d_model,
            nhead=          nhead,
            dns_scale=      dns_scale,
            dropout=        dropout,
            dropout_att=    dropout_att,
            activation=     activation,
            dropout_res=    dropout_res,
            do_zeroes=      self.do_zeroes,
        ) for _ in range(num_layers_to_build)]
        for lix,lay in enumerate(layers): self.add_module(f'lay_{lix}',lay)

        if shared_lays is not None:
            exp_layers = []
            for lay,mul in zip(layers,shared_lays):
                exp_layers += [lay]*mul
            layers = exp_layers

        self.layers = layers[:num_layers]
        self.layers_TAT = layers[num_layers:]

        self.norm = torch.nn.LayerNorm(normalized_shape=self.d_model)

    def _encode(self, inp:TNS, mask:Optional[TNS]=None) -> DTNS:
        """ base Transformer encoding """

        output = inp

        # it is possible to receive higher than 3-dim input [batch,seq,feats]
        # we need to flatten such dim since it is not supported by native torch Transformer:
        # "query should be unbatched 2D or batched 3D tensor but received 4-D query tensor" <- MultiheadAttention(Module)
        orig_shape = output.shape
        if len(orig_shape) > 3:
            output = output.view([-1] + list(orig_shape)[-2:])

        # add positional embeddings if needed
        if self.pos_emb is not None:
            output += self.pos_emb[:output.size(-2)]

        zsL = [] if self.do_zeroes else None
        for mod in self.layers:
            block_out = mod(inp=output, inp_mask=mask)
            output = block_out['out']
            if self.do_zeroes:
                zsL.append(block_out['zeroes'])

        # pass through task-attention layers
        if self.layers_TAT:

            seq = output

            if self.initial_task_query is not None:
                task_query = self.initial_task_query                # from model
                seq_shape = list(seq.shape)
                seq_shape[-2] = 1
                # TODO: not sure if it will work for batch of sequences, e.g. [128,64,512]
                task_query = task_query.view(seq_shape)
            else:
                task_query = torch.mean(seq, dim=-2, keepdim=True)  # first initial from reduce

            for mod in self.layers_TAT:
                block_out = mod(inp=seq, task_query=task_query, inp_mask=mask)
                task_query = block_out['out']
                if self.do_zeroes:
                    zsL.append(block_out['zeroes'])
            output = torch.flatten(task_query,-2,-1) # remove seq (-2) dimension of size 1

        output = self.norm(output)

        # eventually roll back to original shape
        if len(orig_shape) > 3:
            output = output.view(orig_shape)

        if zsL:
            zsL = torch.cat(zsL).detach()

        return {
            'out':      output,
            'zeroes':   zsL}

    def _encode_pyramidal(self, inp:TNS, pyramide: Union[Tuple[int],int]) -> DTNS:
        """ pyramidal_encoding """

        if type(pyramide) is int: pyramide = (pyramide,)

        inp = inp
        zsL = []
        for sl in pyramide:
            in_split = torch.split(inp, sl, dim=0)
            outL = [self._encode(inp=i) for i in in_split]
            inp = [o['out'] for o in outL]
            inp = torch.stack(inp, dim=0)
            zsL += [o['zeroes'] for o in outL]
        out = self._encode(inp)
        zsL.append(out['zeroes'])
        out['zeroes'] = torch.cat(zsL).detach()
        return out

    def forward(
            self,
            inp: TNS,
            mask: Optional[TNS]=                        None,
            pyramide: Optional[Union[Tuple[int],int]]=  None) -> DTNS:

        inp_orig_shape = inp.shape

        # expand single token
        if len(inp_orig_shape) == 1:
            inp = inp.expand([1, inp_orig_shape[0]])

        if pyramide:
            if mask is not None:
                raise TorchnessException('mask is not supported for pyramidal encoding')
            enc_out = self._encode_pyramidal(inp, pyramide)
        else:
            enc_out = self._encode(inp, mask)

        if len(inp_orig_shape) == 1:
            enc_out['out'] = torch.squeeze(enc_out['out'], dim=0)

        return enc_out