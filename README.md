# COCOB-optimizer
PyTorch implementation of COCOB optimizer in [Training Deep Networks without Learning Rates
Through Coin Betting](http://papers.nips.cc/paper/6811-training-deep-networks-without-learning-rates-through-coin-betting) [1]

## Usage
1. Put cocob_bp.py in YOUR_PYTHON_PATH/site-packages/torch/optim.                                                     
2. Open YOUR_PYTHON_PATH/site-packages/torch/optim/\_\_init\_\_.py add the following code:                                
```                                                                                                                   
from .cocob_bp import COCOB_Backprop                                                                                  
del cocob_bp                                                                                                          
```                                                                                                                   
3. Save \_\_init\_\_.py and restart your python.                                                                          
Use COCOB_Backprop as                                                                                                 
```                                                                                                                          
optimizer = optim.COCOB_Backprop(net.parameters())                                                                    
...                                                                                                                   
optimizer.step()               
```
Implemented by Huidong Liu

## Reference
[1] Francesco Orabona and Tatiana Tommasi, Training Deep Networks without Learning Rates Through Coin Betting, NIPS 2017. 
