import numpy as np
import torch.nn as nn
import torch, torch.utils.data
import math 
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

class BatchNormFlow(nn.Module):
    """An implementation of a batch normalization layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros( num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)
                
                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)
        
class CondCouplingLayer(nn.Module):
    """An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803) extended
    to allow optional conditioning.
    """
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=0,
                 act=nn.ReLU):
        super(CondCouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask
        self.num_cond_inputs = num_cond_inputs
        
        self.shared_layers = nn.Sequential(
            nn.Linear(num_inputs+self.num_cond_inputs, num_hidden),
            act(),
            nn.Linear(num_hidden, num_hidden),
            act())
        
        self.scale_net = nn.Linear(num_hidden, num_inputs)
        self.translate_net = nn.Linear(num_hidden, num_inputs)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.orthogonal_(m.weight)
        
        self.shared_layers.apply(_init_weights)
        self.scale_net.apply(_init_weights)
        self.translate_net.apply(_init_weights)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask
        if (self.num_cond_inputs > 0) and (cond_inputs is not None):
            net_inputs = torch.cat((inputs * mask, cond_inputs), -1)
        else:
            net_inputs = inputs * mask
        
        net_output = self.shared_layers(net_inputs)
        log_s = torch.tanh(self.scale_net(net_output)) * (1 - mask)
        t = self.translate_net(net_output) * (1 - mask)

        if mode == 'direct':
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)
    

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device) # changed

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self.forward(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)
    
    def sample(self, nsamps, cond_inputs):
        return self.forward(torch.randn((nsamps, self.num_inputs), device=device), cond_inputs=cond_inputs, mode= 'inverse')

    
class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu'):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = self._get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = self._get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = self._get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)
        
    def _get_mask(self, in_features, out_features, in_flow_features, mask_type=None):
        """
        mask_type: input | None | output

        See Figure 1 for a better illustration:
        https://arxiv.org/pdf/1502.03509.pdf
        """
        if mask_type == 'input':
            in_degrees = torch.arange(in_features) % in_flow_features
        else:
            in_degrees = torch.arange(in_features) % (in_flow_features - 1)

        if mask_type == 'output':
            out_degrees = torch.arange(out_features) % in_flow_features - 1
        else:
            out_degrees = torch.arange(out_features) % (in_flow_features - 1)

        return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output

class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)

class CondFlow(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_blocks, num_cond_inputs, flow_type = "rnvp", emb_cond_inputs = None):
        super(CondFlow, self).__init__()
        self.num_inputs = num_inputs
        assert(num_inputs > 1)
        if emb_cond_inputs is None:
            self.embedding_modules = None
            self.num_cond_inputs = num_cond_inputs
        else:
            self.embedding_modules = []
            self.embedding_limits = []
            self.num_cond_inputs = 0
            for ((start, stop), dict_size, emb_size) in emb_cond_inputs:
                self.embedding_modules.append(nn.Embedding(int(dict_size), int(emb_size)))
                self.embedding_limits.append((start, stop))
                self.num_cond_inputs += emb_size
        
        if flow_type == "rnvp":
            mask = torch.arange(0, num_inputs, device=device) % 2
            mask = mask.float()
            modules = []
            for _ in range(num_blocks):
                modules += [
                    CondCouplingLayer(
                        num_inputs, num_hidden, mask, 
                        num_cond_inputs=self.num_cond_inputs,
                        act=nn.ELU),
                    BatchNormFlow(num_inputs)
                ]
                mask = 1 - mask
            
            modules += [
                    CondCouplingLayer(
                        num_inputs, num_hidden, mask, 
                        num_cond_inputs=self.num_cond_inputs,
                        act=nn.ELU),
            ]
        elif flow_type == 'maf':
            modules = []
            for _ in range(num_blocks):
                modules += [
                    MADE(num_inputs, num_hidden, num_cond_inputs=num_cond_inputs, act='tanh'),
                    BatchNormFlow(num_inputs),
                    Reverse(num_inputs)
            ]
        else:
            assert()
        self.flow = FlowSequential(*modules)
    
    def embed(self, cond):
        embd_cond = cond
        if self.embedding_modules is not None:
            embd_cond = torch.zeros([cond.shape[0],0])
            for i, (start, stop) in enumerate(self.embedding_limits): 
                new_emb = self.embedding_modules[i](cond[:, start:stop]).squeeze()
                if len(new_emb.shape) < 2:
                    new_emb = new_emb.reshape(1, -1)
                embd_cond = torch.cat(( embd_cond, new_emb), -1)
        return embd_cond
        
    def forward(self, cond, observed):
        if len(cond.shape) == 1:
            print("reshaped")
            cond = cond.reshape(1, -1)
        cond = self.embed(cond)
        assert(observed.shape[1] == self.num_inputs)
        assert(cond.shape[1] == self.num_cond_inputs)
        assert(observed.shape[0] == cond.shape[0])
        if len(observed.shape) > len(cond.shape):
            print("assuming multiple outputs per input. Reshaping.")
        neg_log_probs = -self.flow.log_probs(observed, cond)
        
        return neg_log_probs
    
    def log_pdf(self, inp, outp):
        if self.embedding_modules:
            inp = torch.Tensor(inp).long().to(device)
        else:
            inp = torch.Tensor(inp).to(device)
        return -self.forward(inp, torch.Tensor(outp).to(device))
    
    def sample(self, nsamps, inp):
        if self.embedding_modules:
            inp = self.embed(torch.Tensor(np.atleast_2d(inp)).long().to(device))
        else:
            inp = torch.Tensor(inp).to(device)
            
        return self.flow.sample(nsamps, inp.repeat(nsamps,1))[0]
    
    def set_dataset(self, ds, batch_size = 256):
        if batch_size == "Full Dataset":
            batch_size = len(ds)
        self.dl = torch.utils.data.DataLoader(ds, batch_size = batch_size, shuffle=True)
    
    def fit(self, epochs = 1, learning_rate = 5e-4, weight_decay=1e-6):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        l = []
        for i in range(epochs):
            for batch_idx, sample_batched in enumerate(self.dl):
                self.zero_grad()
                loss = self.forward(sample_batched[0], sample_batched[1]).sum()                
                loss.backward()
                optimizer.step()
                #print("%3d" % batch_idx, loss.detach().cpu().numpy())
            #l.append(loss.item())
            print("%3d" % i, loss.detach().cpu().numpy())
        self.eval()
        
    
    @staticmethod
    def optimized_cde(train_x, train_y, num_hidden = 100, num_layers = 5, epochs = 2, learning_rate = 5e-4, weight_decay=1e-6, batch_size = 256, emb_cond_inputs = None, flow_type='rnvp'):
        class dset(torch.utils.data.Dataset):
            def __init__(self, inp, outp):                
                if emb_cond_inputs:
                    self.inp = torch.Tensor(inp).long().to(device)
                else:
                    self.inp = torch.Tensor(inp).to(device)
                self.outp = torch.Tensor(outp).to(device)
            def __len__(self):
                return self.outp.shape[0]
            
            def __getitem__(self, idx):
                return self.inp[idx], self.outp[idx]
        
        cd = CondFlow(train_y.shape[1], num_hidden, num_layers, train_x.shape[1], emb_cond_inputs=emb_cond_inputs, flow_type = flow_type)
#<<<<<<< Updated upstream
#        cd.set_dataset(dset(torch.Tensor(train_x).to(device), torch.Tensor(train_y).to(device)), batch_size)
#=======
        cd.to(device)
        print("Constructing dataset")
        dataset = dset((train_x), (train_y))
        
        cd.set_dataset(dataset, batch_size)
#>>>>>>> Stashed changes
        print("fitting")
        cd.fit(epochs = epochs, learning_rate = learning_rate, weight_decay = weight_decay)
        return cd
    
    @staticmethod
    def test_optimized_cde(flow_type = 'rnvp'):
        import pylab as pl
        x = np.concatenate((np.zeros((1000,1)),np.ones((1000,1))+3),0) 
        y = x + np.random.randn(x.shape[0],2)
        cr = CondFlow.optimized_cde(x, y, epochs=50, batch_size=x.shape[0]#, emb_cond_inputs=[((0,1), 5, 10)]
                                    , flow_type=flow_type)
        
        pl.scatter(*cr.sample(1000,np.array([0])).detach().numpy().T)
        pl.scatter(*cr.sample(1000,np.array([4])).detach().numpy().T)
        pl.scatter(*cr.sample(1000,np.array([2])).detach().numpy().T)
    