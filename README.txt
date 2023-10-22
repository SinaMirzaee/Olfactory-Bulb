Olfactory Bulb v1.00

This neural network model represents the olfactory bulb and olfactory receptors. There are two major populations of neurons in the olfactory bulb; mitral cells (Excitatory population) and granule cells (inhibitory population). The interconnections between these two layers create a system of coupled oscillators, oscillating around an equilibrium point that can be shifted by inputs from olfactory receptors. As a result, the output of the olfactory bulb (mitral cells' output) has a characteristic oscillation pattern for each odor pattern.

An odor pattern would be given to the network as input, along with the moments of exhale-inhale transition. The first layer represents Olfactory receptors, which detect odor molecules. The output of this layer is a vector containing the firing rate of each receptor. 
The second and third layers represent the olfactory bulb. Each of these layers consists of two "sub-layers"; one sub-layer computes the membrane potential of each cell, while the second sub-layer determines the firing rate (output) of each cell.
The output of the first layer is given to the mitral layer by one-to-one connections. The connections between mitral and granule cells mimic the olfactory bulb (Li and Hopfield 1989 provide more information on these connection patterns).

This model was introduced by Li and Hopfield in their article "Modeling the Olfactory Bulb and its Neural Oscillatory Processing (1989)".

I'd like to develop this model further using associative memory to detect odor patterns. I'd be very grateful if you shared your ideas and knowledge on this subject with me, and if you have suggestions which might improve this code, please contact me by email.