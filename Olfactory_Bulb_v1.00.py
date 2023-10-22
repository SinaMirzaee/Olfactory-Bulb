import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


class MembranePotential(nn.Module):
    ''' This "sub-layer" determines how inputs to each neuron would influence its membrane potential. Synapses
        and their strength is also defined in this layer.'''
    
    def __init__(self, cell_type, cell_count) -> None:
        super().__init__()
        self.celltype = cell_type # neurons are either Mitral cells or Granule cells
        self.potential = torch.ones(cell_count,1,) * 0.3 # initial potential 
        
        '''
        H0 and W0 are weight matrices that determine how neurons in a layer are connected to the previous
        one. More information on how these matrices are constructed can be found in the README.txt file. 
        '''
        if cell_type == "Mitral":
            self.H0 = torch.tensor([[0.3, 0.9, 0, 0, 0, 0, 0, 0, 0, 0.7],
                                    [0.9, 0.4, 1.0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0.8, 0.3, 0.8, 0, 0, 0, 0 ,0, 0],
                                    [0, 0, 0.7, 0.5, 0.9, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0.8, 0.3, 0.3, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0.7, 0.3, 0.9, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0.7, 0.4, 0.9, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.7, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0.9, 0.3, 0.9],
                                    [0.7, 0, 0, 0, 0, 0, 0, 0, 0.8, 0.3]])
        elif cell_type == "Granule":
            self.W0 = torch.tensor([[0.3, 0.7, 0, 0, 0, 0, 0, 0, 0.5, 0.3],
                                    [0.3, 0.2, 0.5, 0, 0, 0, 0, 0, 0, 0.7],
                                    [0, 0.1, 0.3, 0.5, 0, 0, 0, 0 ,0, 0  ],
                                    [0, 0.5, 0.2, 0.2, 0.5, 0, 0, 0, 0, 0],
                                    [0.5, 0, 0, 0.5, 0.1, 0.9, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0.3, 0.3, 0.5, 0.4, 0, 0],
                                    [0, 0, 0, 0.6, 0, 0.2, 0.3, 0.5, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0.5, 0.3, 0.5, 0  ],
                                    [0, 0, 0, 0, 0, 0.2, 0, 0.2, 0.3, 0.7],
                                    [0.7, 0, 0, 0, 0, 0, 0, 0.2, 0.3, 0.5]])

    ''' First our model will update the internal state of each neuron based on the inputs it recieves.
    Each neuron layer recieves two types of inputs;

            Mitral cells recieve one input from olfactory receptors (I or the vector input[0]),
        and another from granule cells (V_y or the vector input[1]) which is inhibitory.
        The internal state of each mitral cell (X) changes with the rate:

            X' = -H0 * V_y - 1/tau_x * X + I
        
        where tau_x is the time constant of mitral cells, here we assume tau_x = 7ms. 

            Granule cells recieve one input from higher olfactory centers (I_c or the vector
        input[0]) which we assume to be constant (I_c = 0.1),and another from mitral cells (V_y or the
        vector input[1]).
        The internal state of each mitral cell (X) changes with the rate:

            Y' = W0* V_x - 1/tau_y * Y + I
        
        where tau_y is the time constant of granule cells, here we assume tau_x = 7ms.
    '''           
    def forward(self, input):
        if self.celltype == "Mitral":
            self.potential = self.potential - 2 * ((self.H0@input[1]) + self.potential * 1/7 -  input[0])
        elif self.celltype == "Granule":
            self.potential = self.potential + 2 * ((self.W0@input[1]) - self.potential * 1/7 +  input[0])
        return self.potential


class FiringRate(nn.Module):
    ''' this "sub-layer" determines the output (the firing rate ) of each neuron based on its 
        membrane potential   
    '''
    def __init__(self, cell_type, cell_count) -> None:
        super().__init__()
        self.celltype = cell_type
        self.fire_rate = torch.zeros(cell_count,1,)
    
    def Out(self, potential):
        ''' Firing rate of the neurons are determined by following equations:
            Mitral cells:
                Firing rate = {S'x + S'x * tanh((p - th)/S'x)  when p < th
                               S'x + Sx * tanh((p - th)/Sx)    when p >= th}
                where Sx = 1.4, S'x = 0.14, and th = 1. 
            Granule cells: 
                Firing rate = {S'y + S'y * tanh((p - th)/S'y)  when p < th
                               S'y + Sy * tanh((p - th)/Sy)    when p >= th}
                where Sx = 2.9, S'x = 0.29, and th = 1.
        '''
        i = 0
        if self.celltype == "Mitral":
            for p in potential:
                if p < 1:
                    self.fire_rate[i] = 0.14 + 0.14 * math.tanh((p - 1)/0.14)
                else:
                    self.fire_rate[i] = 0.14 + 1.4 * math.tanh((p - 1)/1.4)
                i+=1
        elif self.celltype == "Granule":
            for p in potential:
                if p < 1:
                    self.fire_rate[i] = 0.29 + 0.29*math.tanh((p - 1)/0.29)
                else:
                    self.fire_rate[i] = 0.29 + 2.9*math.tanh((p - 1)/2.9)
                i+=1


class Neuron_Layer(nn.Module):
    ''' This layer represents the neurons in the olfactory bulb; a layer if mitral or granule cells,
        which consists of two "sub-layer" that each determines membrane potentials and firing rate of 
        each cell. These sub-layers has been defined above.
    '''
    def __init__(self, cell_type, cell_count) -> None:
        super().__init__()
        self.potential = MembranePotential(cell_type, cell_count)
        self.fire_rate = FiringRate(cell_type, cell_count)

    def forward(self, input):
        potential = self.potential.forward(input)
        self.fire_rate.Out(potential)
        #return self.fire_rate.fire_rate

class Receptors(nn.Module):
    ''' This layer represents the olfactory receptors. Odor patterns are used as input to this layer. Also 
        breathing pattern is given as input. The output of this layer (I) would activates mitral cells. '''
    def __init__(self, cell_count) -> None:
        super().__init__()
        self.cell_count = cell_count # Number of receptor cells in this layer
        self.I = torch.ones((cell_count,1,)) * 0.243

    def forward(self ,input, t, period, Inhale) -> torch.Tensor:
        if Inhale:
            self.I = self.I + input[0] * 7
            return self.I
        else:
            self.I = self.I - (7 * self.I/33 * math.e**((input[1][period - 1]-t)/33))
            #self.I.sub_(torch.mul(self.I, input[1][period] * 1/33 * math.e**((input[1][period]-t)/33)))
            return self.I

class OlfactoryBulb(nn.Module):
    ''' This is our model of olfactory bulb and olfactory receptors. This neural circuit consists of 
        three layers; Olfactory receptors, excitatory mitral cells and inhibitory granule cells. 

    '''
    def __init__(self, cell_count) -> None:
        super().__init__()
        self.Receptor = Receptors(cell_count)
        self.Mitral = Neuron_Layer('Mitral', cell_count)
        self.Granule = Neuron_Layer('Granule', cell_count)
    
    def forward(self, input, t, period, Inhale):
        I = self.Receptor.forward(input, t, period, Inhale)
        I_c = torch.ones(10,1,) * 0.1
        self.Mitral.forward([I, self.Granule.fire_rate.fire_rate])
        self.Granule.forward([I_c, self.Mitral.fire_rate.fire_rate])
        
model = OlfactoryBulb(10)
data = [[0.002, 0.001, 0.001, 0, 0.00152, 0.00152, 0.0001, 0, 0.003, 0.0003],[370, 570, 770, 970, 1170, 1300, 1506, 1700, 1962, 2100]]
odor_pattern = torch.tensor(data[0]).reshape([10,1])
input = [odor_pattern,data[1]] #inputs are odor pattern and moments of inhale-exhale transition

out_rec_5 = []
out_rec_8 = []
granule_rec_0 = []
I_rec_0 = []
time = []

for t in range(0,2000,7):
    for i in range(len(data[1])):
        if t <= data[1][i]:
            period = i
            break
    if period % 2 == 0:
        inhale = True
    else:
        inhale = False

    model.forward(input, t, period, inhale)

    out_rec_5.append(model.Mitral.fire_rate.fire_rate[5].item())
    out_rec_8.append(model.Mitral.fire_rate.fire_rate[8].item())
    granule_rec_0.append(model.Granule.fire_rate.fire_rate[0].item())
    I_rec_0.append(model.Receptor.I[5].item())
    time.append(t)

plt.plot(time, out_rec_5)
plt.plot(time, out_rec_8)
plt.plot(time, granule_rec_0)
#plt.plot(time, I_rec_0)
plt.show()
    