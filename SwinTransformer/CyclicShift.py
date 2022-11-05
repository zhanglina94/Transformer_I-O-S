

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
    	# x.shape (b, 56, 56, 96)
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))  
