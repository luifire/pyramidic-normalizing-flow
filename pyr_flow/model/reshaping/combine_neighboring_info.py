import torch.nn as nn

from constants import *
from model.layer_module import LayerModule

from model.reshaping.initial_reshaping import merge_patches

""" 
Without batch

[[[a,b,c], [d,e,f]]
[g,h,i],[j,k,l]]
shape h * w * 3
->
[[ [a,d,g,j], [b,e,h,k], [c,f,i,l] ]]
h/2 * w/2 * 3 * 4

-> # later:
a := a'
use the most high level information
[[ [a,d, b,e, c,f] ]
h/2 * w/2 * (4/2) * 3
"""

"""
Problem: was passiert mit Pixel, die keine Nachbarn mehr haben?
1. sie werden komplett ignoriert und kommen nicht mal mehr in die Berechnung 
des loss mit rein.
(Bild von 33 Pixel: 
33 -> 11x11
-> 10x10 (verliere: 11*2*3) Pixel
-> 5x5
-> 4x4 (verliere 5*2* (kp zu viele)
...

2. special treatment für die
kommen in einen Waste Bin

=>
Die kommen in einen Waste Bin
-> bekommen dann nochmal eine Verarbeitung
-> werden dann gleich gewichtet wie der Rest, 
der so mit denen raus fällt

-> man könnte auch das Zeug links davon weiter verarbeiten und dann nochmal 
mit dem Waste kombinieren.
Dadurch bekommt es nochmal high level info, darf aber nichts mehr zur Spitze beitragen
"""
#TODO siehe Kommentar oben drüber


class CombineNeighbors(LayerModule):

    def __init__(self):
        super().__init__()

        # dummy registration
        self.dummy = nn.Parameter(torch.zeros(1, device=DEVICE), requires_grad=False)

        print(f'Combine Neighbors - {COMBINE_NEIGHBOR_KERNEL_SIZE}x{COMBINE_NEIGHBOR_KERNEL_SIZE}')

    """Removes overlapping pixel on the right side
    (we need to be devisible by 2) """
    def _seperate_waste(self, x):
        _, h, w, depth = x.shape
        assert h == w
        if h % 2 == 0:
            return x, None

        waste_bottom = x[:,-1,:,:]
        waste_right = x[:,0:-1,-1,:]

        waste = torch.cat([waste_bottom, waste_right], 1)

        x = x[:,:-1,:-1,:]

        return x, waste


    def forward(self, x):
        x, waste = self._seperate_waste(x)

        x = merge_patches(x, COMBINE_NEIGHBOR_KERNEL_SIZE)

        return x, waste

