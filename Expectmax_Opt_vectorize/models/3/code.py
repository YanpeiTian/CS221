class NN2048(nn.Module):
    def __init__(self, input_size=16, filter1=256, filter2=4096, drop_prob=0.):
        super(NN2048, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=(1,1), padding=0)
        self.conv_2 = nn.Conv2d(in_channels=input_size, out_channels=input_size, kernel_size=(1,1), padding=0)
        self.conv_3 = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,1), padding=0)
        self.conv_4 = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,1), padding=0)
        
        self.conv_a = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,1), padding=0)
        self.conv_b = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(1,2), padding=0)
        self.conv_c = nn.Conv2d(in_channels=input_size, out_channels=filter1, kernel_size=(2,2), padding=0)
        
        self.conv_aa = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_ab = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        self.conv_ba = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,1), padding=0)
        self.conv_bb = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(1,2), padding=0)
        
        self.conv_c2 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(2,2), padding=0)
        self.pool = nn.MaxPool2d(2)
        
        self.relu = nn.ReLU()
        self.W_aa = nn.Linear(filter2 * 8, 1)
        self.W_ab = nn.Linear(filter2 * 9, 1)
        self.W_ba = nn.Linear(filter2 * 9, 1)
        self.W_bb = nn.Linear(filter2 * 8, 1)
        self.W_c = nn.Linear(filter2 * 1, 1)
        self.W_3 = nn.Linear(filter1 * 4, 1)
        self.W_4 = nn.Linear(filter1 * 16, 1)

    def flatten(self, x):
        N = x.size()[0]
        return x.view(N, -1)
        
    def forward(self, x):
        x = x.float()
        x1 = self.relu(self.conv_1(x))
        x2 = self.relu(self.conv_2(x))
        x3 = self.flatten(self.relu(self.conv_3(self.pool(x))))
        x4 = self.flatten(self.relu(self.conv_4(x)))
        
        a = self.relu(self.conv_a(x1))
        b = self.relu(self.conv_b(x1))
        c = self.relu(self.conv_c(x2))

        aa = self.flatten(self.relu(self.conv_aa(a)))
        ab = self.flatten(self.relu(self.conv_ab(a)))
        ba = self.flatten(self.relu(self.conv_ba(b)))
        bb = self.flatten(self.relu(self.conv_bb(b)))
        
        c2 = self.relu(self.conv_c2(c))
        c3 = self.flatten(self.pool(c2))
        
        out = self.W_aa(aa) + self.W_ab(ab) + self.W_ba(ba) + self.W_bb(bb) + \
              self.W_c(c3) + self.W_3(x3) + self.W_4(x4)
        
        return out


lr = 1e-4
weight_decay = 1e-6
beta1 = 0.9

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
        if epoch % 500 == 0:
            filename = "model3_"+str(epoch)+".pth.tar"
            save_model({
                'epoch': epoch,
                'state_dict': model.cpu().state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename)
            model.cuda()
    
train(model, optimizer, loss)


num_epochs = 100

def test(model):
    epoch = 0
    while epoch != num_epochs:
        epoch += 1
        game_len, max_score, game_score, last_loss = gen_sample_and_learn(model, None, None, False)
        print ('epoch', epoch, game_len, max_score, game_score, last_loss)

test(model)