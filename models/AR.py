import torch
import torch.nn as nn
import torch.nn.functional as F

def pixelcnn_gate(x):
    a, b = x.chunk(2,1)
    return torch.tanh(a) * torch.sigmoid(b)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type=None, mask_n_channels=None, gated=False, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(self.bias, 0.)
        self.mask_type = mask_type
        self.mask_n_channels = mask_n_channels
        center_row = self.kernel_size[0] // 2
        center_col = self.kernel_size[1] // 2
        
        mask = torch.ones_like(self.weight)
        
        mask[:, :, center_row + 1:,:] = 0 
        mask[:, :, center_row, center_col + 1:] = 0
        if mask_type == 'vstack':
            mask[:, :, center_row, :] = 0
        elif mask_type == 'a':
            mask[:, :, center_row, center_col] = 0
        #elif mask_type == 'b':
        #    pass
        if gated:
            mask = mask.chunk(2,0)[0].repeat(2,1,1,1)
        # final mask
        self.register_buffer('mask', mask)

    def forward(self, x):
        #print( 'mask_n_channels', self.mask_n_channels)
        #print('kernel shape', self.kernel_size)
        #print('data shape',self.weight.data.shape )
        #print(self.mask_type, self.mask.shape, self.mask)
        #if self.gated :
        #    print(self.gated)
        self.weight.data *= self.mask
        return super().forward(x)


class GatedResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, mask_n_channels, norm_layer):
        super().__init__()
        self.residual = (in_channels==out_channels)
        self.norm_layer = norm_layer

        self.v   = MaskedConv2d(in_channels, 2*out_channels, kernel_size, padding=kernel_size//2,
                                mask_type='vstack', mask_n_channels=mask_n_channels, gated=True)
        
        self.h   = MaskedConv2d(in_channels, 2*out_channels, (1, kernel_size), padding=(0, kernel_size//2),
                                mask_type=mask_type, mask_n_channels=mask_n_channels, gated=True)
        
        self.v2h = MaskedConv2d(2*out_channels, 2*out_channels, kernel_size=1,
                                mask_type=mask_type, mask_n_channels=mask_n_channels, gated=True)
        
        self.h2h = MaskedConv2d(out_channels, out_channels, kernel_size=1,
                                mask_type=mask_type, mask_n_channels=mask_n_channels, gated=False)


        if self.norm_layer:
            self.norm_layer_v = nn.BatchNorm2d(out_channels)
            self.norm_layer_h = nn.BatchNorm2d(out_channels)

    def forward(self, x_v, x_h):
        # vertical stack
        # Fill this 
        x_v = self.v( x_v )
        #print('gate-1')
        x_v_out = pixelcnn_gate( x_v )
        x_v = self.v2h( x_v )
        #print('x_v',x_v.shape)
        #print('x_v_out',x_v_out.shape)
        #print('gate-2')
        # horizontal stack
        # Fill this 
        x_h_out = pixelcnn_gate(self.h( x_h )+x_v)
        #print('gate-3')
        x_h_out = self.h2h( x_h_out )
        #print('gate-4')
        # residual connection
        if self.residual:
            x_h_out = x_h_out + x_h

        # normalization
        if self.norm_layer:
            x_v_out = self.norm_layer_v(x_v_out) 
            x_h_out = self.norm_layer_h(x_h_out) 

        return x_v_out, x_h_out




# --------------------
# PixelCNN
# --------------------

class PixelCNN(nn.Module):
    def __init__(self, in_channels, n_bits, hidden_dim, output_dim, kernel_size, n_res_layers, norm_layer=True):
        super().__init__()
        C = in_channels

        self.n_bits = n_bits
        self.input_conv = MaskedConv2d(C, 2*hidden_dim, kernel_size=7, padding=3, mask_type='a', mask_n_channels=C, gated=True)
        self.res_layers = nn.ModuleList([
            GatedResidualLayer(hidden_dim, hidden_dim, kernel_size, 'b', C, norm_layer)
            for _ in range(n_res_layers)])
        self.conv_out1 = MaskedConv2d(hidden_dim, 2*output_dim, kernel_size=1, mask_type='b', mask_n_channels=C, gated=True)
        self.conv_out2 = MaskedConv2d(output_dim, 2*output_dim, kernel_size=1, mask_type='b', mask_n_channels=C, gated=True)
        self.output = MaskedConv2d(output_dim, C * 2**n_bits, kernel_size=1, mask_type='b', mask_n_channels=C)

    def forward(self, x, h=None):
        B, C, H, W = x.shape
        #print('--1--')
        x = pixelcnn_gate(self.input_conv(x))
        #print('--2--')
        x_v, x_h = x, x
        for l in self.res_layers:
            x_v, x_h = l(x_v, x_h)
        #print('--3--')

        out = pixelcnn_gate(self.conv_out1(x_h))
        #print('--4--')
        out = pixelcnn_gate(self.conv_out2(out))
        #print('--5--')
        out = self.output(out)

        return out.reshape(B, -1, C, H, W)

