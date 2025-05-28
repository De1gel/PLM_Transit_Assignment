# A strategy-based model for comparison
## Line and node strategy-based model
this model is proposed by the paper "A new transit assignment model based on line and node strategies" by Ren et al. in 2021. The result of this model has been proven to be equivalent to the basic strategy model (please refer to the paper for more details), so the results of this model can be used to represent the results of the basic strategy model for comparison. If this implementation infringes on anyone's intellectual property rights, please contact me to delete it.

## Usage
This model is implemented in Python and can be run using the following functions:

```python
net = Network()  # Create a network instance
lns = LNS(2)  # Create a Line and Node Strategy instance with a given max transfer time (2 is enough to cover this network)
msa_lns = MSA_LNS(0.001, 100)
msa_lns.Method_of_Successive_Algorithm(net, 2, lns)
```
