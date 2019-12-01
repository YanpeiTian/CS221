class NN2048(nn.Module):
    def __init__(self, input_size=16, filter1=64, filter2=256, drop_prob=0.):
        super(NN2048, self).__init__()
        self.conv_a = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,1), padding=0)
#         self.conv_a3 = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(3,1), padding=0)
        self.conv_a4 = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(4,1), padding=0)
        self.conv_b = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,2), padding=0)
#         self.conv_b3 = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,3), padding=0)
        self.conv_b4 = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,4), padding=0)
        self.conv_c = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,2), padding=0)
        
#         self.conv_a = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,1), padding=0)
#         self.conv_b = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,2), padding=0)
        self.conv_aa = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_ab = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.conv_ba = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_bb = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        
        self.conv_ab4 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,4), padding=0)
        self.conv_ba4 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(4,1), padding=0)
        self.conv_c2 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,2), padding=0)
        self.pool = nn.MaxPool2d(2)
        
        self.relu = nn.ReLU()
        self.W_aa = nn.Linear(filter2 * 8, 1)
        self.W_ab = nn.Linear(filter2 * 9, 1)
        self.W_ba = nn.Linear(filter2 * 9, 1)
        self.W_bb = nn.Linear(filter2 * 8, 1)
        self.W_ab4 = nn.Linear(filter2 * 1, 1)
        self.W_ba4 = nn.Linear(filter2 * 1, 1)
        self.W_c = nn.Linear(filter2 * 1, 1)

    def flatten(self, x):
        N = x.size()[0]
        return x.view(N, -1)
        
    def forward(self, x):
        x = x.float()
        a = self.relu(self.conv_a(x))
        b = self.relu(self.conv_b(x))
        c = self.relu(self.conv_c(x))
        a4 = self.relu(self.conv_a4(x))
        b4 = self.relu(self.conv_b4(x))
        
        aa = self.flatten(self.relu(self.conv_aa(a)))
        ab = self.flatten(self.relu(self.conv_ab(a)))
        ba = self.flatten(self.relu(self.conv_ba(b)))
        bb = self.flatten(self.relu(self.conv_bb(b)))
        ab4 = self.flatten(self.relu(self.conv_ab4(a4)))
        ba4 = self.flatten(self.relu(self.conv_ba4(b4)))
        c2 = self.relu(self.conv_c2(c))
        c3 = self.flatten(self.pool(c2))
        
        out = self.W_aa(aa) + self.W_ab(ab) + self.W_ba(ba) + self.W_bb(bb) + \
              self.W_ab4(ab4) + self.W_ba4(ba4) + self.W_c(c3)
        return out


lr = 1e-3
weight_decay = 1e-6
model = NN2048().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.5, 0.999))
loss_fn=nn.MSELoss()

for j in range(200):
    result = gen_sample_and_learn(model, optimizer, loss_fn)
    print(j, result)
    if result is not None and result[1] >= 4096:
        break


