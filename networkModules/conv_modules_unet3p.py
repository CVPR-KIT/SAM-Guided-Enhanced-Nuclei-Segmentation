import torch
import torch.nn as nn
import torch.nn.functional as F
from networkModules.init_weights import init_weights
import numpy as np


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1, act='relu'):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                if act == 'relu':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                if act == 'GLU':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size*2, ks, s, p),
                                         nn.BatchNorm2d(out_size*2),
                                         nn.GLU(dim=1), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                if act == 'relu':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                if act == 'GLU':
                    conv = nn.Sequential(nn.Conv2d(in_size, out_size*2, ks, s, p),
                                         nn.GLU(dim=1), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class BlurPool2D(nn.Module):
    def __init__(self, kernel_size, stride):
        super(BlurPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.weight = self._get_weights(kernel_size)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        return F.conv2d(x, self.weight, stride=self.stride, groups=x.size(1))

    def _get_weights(self, kernel_size):
        # create a 1D kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size))
        # normalize the kernel to make it a mean filter
        kernel /= kernel_size ** 2
        # expand dimensions to match the input tensor
        kernel = kernel.expand((3, -1, -1, -1))
        return kernel
    
class MaxBlurPool2D_0(nn.Module):
    def __init__(self, kernel_size, stride, in_channels):
        super(MaxBlurPool2D_0, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        #print(f"inside maxblurpool2d: c={in_channels}, k={kernel_size}")
        self.weight = self._get_weights(kernel_size, in_channels)


    def forward(self, x):
        x = self.maxpool(x)  # Apply max-pooling
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        weight = self.weight.to(x.device)
        return F.conv2d(x, weight, stride=1, groups=x.size(1))  # Apply blur

    def _get_weights(self, kernel_size, num_channels):
        # create a 1D kernel
        kernel = torch.ones((num_channels, 1, kernel_size, kernel_size))
        # normalize the kernel to make it a mean filter
        kernel /= kernel_size ** 2
        return kernel

class MaxBlurPool2d_old(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Max pooling
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        # Blurring kernel
        blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        blur_kernel = blur_kernel.view(1, 1, 3, 3)
        blur_kernel = blur_kernel / blur_kernel.sum()
        blur_kernel = blur_kernel.repeat(x.size(1), 1, 1, 1).to(x.device)

        # Applying the blur using the 'depthwise' convolution
        x = F.conv2d(x, blur_kernel, groups=x.size(1), padding=1)

        return x
    
class MaxBlurPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxBlurPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize blur kernel
        self.blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        self.blur_kernel = self.blur_kernel.view(1, 1, 3, 3) / 16.0

    def forward(self, x):
        # Max pooling
        x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # Move blur kernel to the same device as the input
        blur_kernel = self.blur_kernel.to(x.device).repeat(x.size(1), 1, 1, 1)
        
        # Apply blurring
        x = F.conv2d(x, blur_kernel, groups=x.size(1), padding=1)
        
        return x




class MaxBlurPool(nn.Module):
    def __init__(self, in_channels):
        super(MaxBlurPool, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.blur = BlurPool(in_channels, filt_size=4, stride=2)

    def forward(self, x):
        return self.blur(self.max_pool(x))


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer



class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training:
            return x

        # Generate random tensor for DropBlock mask
        random_tensor = torch.rand(x.shape[0], x.shape[1], x.shape[2] - self.block_size + 1, x.shape[3] - self.block_size + 1, device=x.device)
        random_tensor += self.keep_prob
        mask = torch.floor(random_tensor).type(torch.bool)

        # Calculate counts for each block
        count_ones = F.conv2d(mask.float(), torch.ones((1, 1, self.block_size, self.block_size)).to(x.device), stride=1, padding=0)
        
        # Generate block mask
        block_mask = 1 - F.pad(count_ones, (self.block_size // 2, self.block_size // 2, self.block_size // 2, self.block_size // 2))

        # Apply block mask to input
        x = x * block_mask

        # Normalize feature map to keep activation sum unchanged
        x = x * block_mask.numel() / block_mask.sum()
        
        return x