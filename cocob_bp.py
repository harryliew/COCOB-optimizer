from .optimizer import Optimizer

class COCOB_Backprop(Optimizer):
    """Implementation of the COCOB algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    
    Usage:
    1. Put cocob_bp.py in YOUR_PYTHON_PATH/site-packages/torch/optim. 
    2. Open YOUR_PYTHON_PATH/site-packages/torch/optim/__init__.py add the following code:
    ```
    from .cocob_bp import COCOB_Backprop
    del cocob_bp
    ```
    3. Save __init__.py and restart your python. 
    Use COCOB_Backprop as

    optimizer = optim.COCOB_Backprop(net.parameters())
    ...
    optimizer.step()

    Implemented by Huidong Liu
    Email: huidliu@cs.stonybrook.edu or h.d.liew@gmail.com

    References
    [1] Francesco Orabona and Tatiana Tommasi, Training Deep Networks without Learning Rates
    Through Coin Betting, NIPS 2017.
    
    """
    def __init__(self, params, weight_decay=0, alpha=100):
        defaults = dict(weight_decay=weight_decay)
        super(COCOB_Backprop, self).__init__(params, defaults)
        # COCOB initializaiton
        self.W1 = []
        self.W_zero = []
        self.W_one = []
        self.L = []
        self.G = []
        self.Reward = []
        self.Theta = []
        self.numPara = 0
        self.weight_decay = weight_decay
        self.alpha = alpha
        
        for group in self.param_groups:
            for p in group['params']:
                self.W1.append(p.data.clone())
                self.W_zero.append(p.data.clone().zero_())
                self.W_one.append(p.data.clone().fill_(1))
                self.L.append(p.data.clone().fill_(1))
                self.G.append(p.data.clone().zero_())
                self.Reward.append(p.data.clone().zero_())
                self.Theta.append(p.data.clone().zero_())                
                self.numPara = self.numPara + 1     
            

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        pind = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data + self.weight_decay*p.data
                self.L[pind] = self.L[pind].max(grad.abs())
                self.G[pind] = self.G[pind] + grad.abs()
                self.Reward[pind] = (self.Reward[pind]-(p.data-self.W1[pind]).mul(grad)).max(self.W_zero[pind])
                self.Theta[pind] = self.Theta[pind] + grad
                Beta = self.Theta[pind].div( (self.alpha*self.L[pind]).max(self.G[pind]+self.L[pind]) ).div(self.L[pind])
                p.data = self.W1[pind] - Beta.mul(self.L[pind] + self.Reward[pind])
                pind = pind + 1
                
        return loss
    
        
