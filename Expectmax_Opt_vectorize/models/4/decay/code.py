class NN2048(nn.Module):
    def __init__(self, input_size=16, filter1=512, filter2=4096, drop_prob=0.):
        super(NN2048, self).__init__()
        self.conv_a = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,1), padding=0)
        self.conv_b = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,2), padding=0)
        self.conv_aa = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_ab = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.conv_ba = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_bb = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.relu = nn.ReLU()
        self.W_aa = nn.Linear(filter2 * 8, 1)
        self.W_ab = nn.Linear(filter2 * 9, 1)
        self.W_ba = nn.Linear(filter2 * 9, 1)
        self.W_bb = nn.Linear(filter2 * 8, 1)

    def flatten(self, x):
        N = x.size()[0]
        return x.view(N, -1)
        
    def forward(self, x):
        x = x.float()
        a = self.relu(self.conv_a(x))
        b = self.relu(self.conv_b(x))
        aa = self.flatten(self.relu(self.conv_aa(a)))
        ab = self.flatten(self.relu(self.conv_ab(a)))
        ba = self.flatten(self.relu(self.conv_ba(b)))
        bb = self.flatten(self.relu(self.conv_bb(b)))
        out = self.W_aa(aa) + self.W_ab(ab) + self.W_ba(ba) + self.W_bb(bb)
        return out


lr = 1e-3
weight_decay = 1e-6
beta1 = 0.8

model = NN2048().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, 0.999))
loss=nn.MSELoss()

num_epochs = 5000

def train(model, optimizer, loss):
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        game_len, max_score, game_score, last_loss = gen_sample_and_learn(model, optimizer, loss, True, 0)
        print ('epoch', epoch, game_len, max_score, game_score, last_loss)
    
train(model, optimizer, loss)