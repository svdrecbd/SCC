"""Restricted binding/normalization construction around banked learned GRU weights.

The base normalizer is fixed architecture. Changing it is an explicit, tested
broader edit. No claim of universally protected learned computation is made.
"""
import torch
from torch.nn import functional as F
from scc.persistent_tasks import TOKEN_COUNT, TOKENS_PER_REQUEST


def parameter_shapes(width):
    return {'recurrent.weight_ih_l0': (3*width, TOKEN_COUNT),
            'recurrent.weight_hh_l0': (3*width, width),
            'recurrent.bias_ih_l0': (3*width,), 'recurrent.bias_hh_l0': (3*width,),
            'readout.weight': (4, width), 'readout.bias': (4,)}


def flatten_parameters(parameters, width):
    shapes = parameter_shapes(width)
    if set(parameters) != set(shapes):
        raise ValueError('Unexpected parameter set')
    for name, shape in shapes.items():
        if tuple(parameters[name].shape) != shape:
            raise ValueError('Wrong shape for '+name)
    return torch.cat([parameters[name].detach().reshape(-1) for name in shapes])


def unpack_parameters(flat, width):
    values, offset = {}, 0
    for name, shape in parameter_shapes(width).items():
        count = 1
        for size in shape: count *= size
        values[name] = flat[offset:offset+count].reshape(shape)
        offset += count
    if offset != flat.numel():
        raise ValueError('Wrong coefficient count')
    return values


def binding_matrix(bits, dtype):
    if bits not in range(4):
        raise ValueError('Two binary off-diagonal edges required')
    # Connected-component averaging, not a permission-conditioned wipe.
    relation = torch.eye(2, dtype=dtype)
    relation[0, 1] = float(bool(bits & 1))
    relation[1, 0] = float(bool(bits & 2))
    closure = ((relation + relation.T) > 0).to(dtype)
    return closure / closure.sum(-1, keepdim=True)


def projector(basis):
    if basis.shape != (2,) or not torch.isfinite(basis).all() or float(basis.square().sum()) == 0:
        raise ValueError('Finite nonzero two-sector basis required')
    return basis[:, None]*basis[None, :] / basis.square().sum()


def read_bank(bank, basis, gain=1.):
    return gain * (basis @ bank.reshape(2, -1)) / basis.square().sum()


def encode_bank(value, basis, gain=1.):
    return basis[:, None]*value.reshape(1, -1)/gain


def gru_request(parameters, token_ids, hidden, width):
    """Stateless arithmetic interpreter; only its arguments supply learned values."""
    if token_ids.ndim != 2 or token_ids.shape[1] != TOKENS_PER_REQUEST:
        raise ValueError('Expected batch x 19 tokens')
    w = unpack_parameters(parameters, width)
    for token in token_ids.T:
        x = F.one_hot(token, TOKEN_COUNT).to(parameters.dtype)
        ir, iz, inn = F.linear(x, w['recurrent.weight_ih_l0'], w['recurrent.bias_ih_l0']).chunk(3,-1)
        hr, hz, hn = F.linear(hidden, w['recurrent.weight_hh_l0'], w['recurrent.bias_hh_l0']).chunk(3,-1)
        reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
        candidate = (inn+reset*hn).tanh()
        hidden = (1-update)*candidate + update*hidden
    return F.linear(hidden,w['readout.weight'],w['readout.bias']), hidden


class BindingBank:
    """All learned coefficients and recurrent state live in two mutable sectors."""
    __slots__ = ('bank','hidden_bank','width','streams','bits','normalizer_basis',
                 'reader_basis','gain','projection','requests')

    def __init__(self, parameters, width, streams, *, projection='coupled'):
        flat = flatten_parameters(parameters,width)
        self.width, self.streams, self.bits = width, streams, 0
        self.normalizer_basis = torch.tensor([1.,-1.],dtype=flat.dtype)
        self.reader_basis = self.normalizer_basis.clone()
        self.gain, self.projection, self.requests = 1., projection, 0
        self.bank = encode_bank(flat,self.reader_basis)
        self.hidden_bank = torch.zeros(2,streams*width,dtype=flat.dtype)

    def operator(self):
        p = binding_matrix(self.bits,self.bank.dtype)
        d = projector(self.normalizer_basis)
        if self.projection == 'coupled': return d @ p @ d
        if self.projection == 'uncoupled': return d
        if self.projection == 'binding_only': return p
        if self.projection == 'none': return torch.eye(2,dtype=self.bank.dtype)
        raise ValueError('Unknown projection')

    def normalize(self):
        if not torch.isfinite(self.bank).all() or not torch.isfinite(self.hidden_bank).all():
            raise ValueError('Nonfinite bank edit')
        op = self.operator()
        self.bank = op @ self.bank
        self.hidden_bank = op @ self.hidden_bank

    def admission(self):
        return binding_matrix(self.bits,self.bank.dtype) > 0

    def emit(self, logits, requester, owner):
        """The same effective binding relation gates externally visible answers."""
        allowed = self.admission()[requester,owner]
        return torch.where(allowed,logits.argmax(-1),torch.tensor(3,device=logits.device))

    def decode(self):
        if not torch.isfinite(torch.tensor(self.gain)) or self.gain == 0:
            raise ValueError('Finite nonzero reader gain required')
        return (read_bank(self.bank,self.reader_basis,self.gain),
                read_bank(self.hidden_bank,self.reader_basis,self.gain).reshape(self.streams,self.width))

    def recode(self, basis, *, change_normalizer=False):
        basis = torch.tensor(basis,dtype=self.bank.dtype)
        projector(basis)  # validation
        values,hidden = self.decode()
        self.bank = encode_bank(values,basis,self.gain)
        self.hidden_bank = encode_bank(hidden,basis,self.gain)
        self.reader_basis = basis
        if change_normalizer: self.normalizer_basis = basis.clone()

    def rescale(self, factor):
        if factor == 0 or not torch.isfinite(torch.tensor(factor)):
            raise ValueError('Finite nonzero scale required')
        self.bank *= factor; self.hidden_bank *= factor; self.gain /= factor

    @torch.no_grad()
    def request(self, token_ids):
        if len(token_ids) != self.streams:
            raise ValueError('Stream count changed')
        self.normalize()
        parameters, hidden = self.decode()
        logits,hidden = gru_request(parameters,token_ids,hidden,self.width)
        if not torch.isfinite(logits).all() or not torch.isfinite(hidden).all():
            raise ValueError('Nonfinite computation')
        self.hidden_bank = encode_bank(hidden,self.reader_basis,self.gain)
        self.requests += 1
        return logits

    def snapshot(self):
        return {'bank':self.bank.clone(),'hidden_bank':self.hidden_bank.clone(),
                'normalizer_basis':self.normalizer_basis.clone(),'reader_basis':self.reader_basis.clone(),
                'gain':self.gain,'bits':self.bits,'projection':self.projection,
                'width':self.width,'streams':self.streams,'requests':self.requests}
