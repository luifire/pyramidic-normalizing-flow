"""
def cut_data(data):
  #printt("list", list(data.shape))
  #at = list(data.shape)[2] % KERNEL_SIZE
  at = data.shape[2] % KERNEL_SIZE
  return data[:, :, at:, at:]
"""


"""
früheres weight matrix erzeugung

#printt("weights", diag)
        #printt("diag", diag)
        normal_diag_matrix = torch.diag_embed(diag)
        #printt("normal_diag_matrix", normal_diag_matrix)

        zero_diag_matrix = weights - normal_diag_matrix
        #print("zero_diag_matrix", zero_diag_matrix)

        # exp(diag)+1
        #exped_diag = torch.exp(diag) + 1
        warn("currently, diag could become 0")
        exped_diag = diag
        # printt("exped_diag", exped_diag)
        conv_matrix = zero_diag_matrix + torch.diag_embed(exped_diag)
        #printt("conv_matrix", conv_matrix)

        # printt("x", x.shape)
        conv_matrix = conv_matrix#.unsqueeze(0).unsqueeze(0)
        #printt("conv_matrix", conv_matrix.shape)

        det = torch.prod(exped_diag)

        if torch.isnan(det).any():
            print("asdf")

        assert det.item() != 0
        #printt("det", det.item())
        #printt("dev ende", conv_matrix.device)

        return conv_matrix, det



"""


""""
erste conv action 
        # x.shape == batch, channel, width, height
        # do the convolution myself
        # first to the right and then down
        # x.narrow(input, dim, start, length)
        #printt("x", x)
        #duplicate = torch.zeros_like(x)
        new_x = torch.empty_like(x)
        for h in range(height // KERNEL_SIZE):
            row = x.narrow(HEIGHT_DIM, h*KERNEL_SIZE, KERNEL_SIZE)
            for w in range(width // KERNEL_SIZE):
                patch = row.narrow(WIDTH_DIM, w*KERNEL_SIZE, KERNEL_SIZE)
                #printt("patch", patch)
                flattened = patch.reshape((batch_size, channel_count, self.kernel_size_sq))

                new_patch = torch.matmul(flattened, conv_matrix)

                #unflattened =
                new_x[:,:, h*KERNEL_SIZE:(h+1)*KERNEL_SIZE, w*KERNEL_SIZE:(w+1)*KERNEL_SIZE] = new_patch
                #printt("flattened", flattened)
                #rint(str(w) + "-" + str(h))
                #print(patch.shape)
                #print(patch)
                #duplicate[:,:,h*KERNEL_SIZE:(h+1)*KERNEL_SIZE, w*KERNEL_SIZE:(w+1)*KERNEL_SIZE] = patch
                #exit(1)
            #print(duplicate)
            #exit(1)

        #a = x - duplicate
        #printt("sum", a.sum())
        # amount of convs fits

        new_x = new_x * norm
        return new_x
        #exit(1)

        # printt("norm", norm)
        #x = x.view(-1, self.kernel_size*self.kernel_size, x.shape[2] // KERNEL_SIZE, x.shape[3] // KERNEL_SIZE)

        #y = F.conv2d(x, conv_matrix, stride=self.kernel_size)
        #print("y", y.shape)
        #normalization term
        #y = y / norm

        #print(y.requires_grad)
        #return y
        return None
        """
