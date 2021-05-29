import torch.nn.functional as F
import torch

def att_tensor2conv_tensor(x, dim_feature):
    return torch.stack(x.split(dim_feature, dim=-1), dim=1)

def conv_tensor2att_tensor(x):
    return dim_combine(x.permute(0, 2, 1, 3), 2, 4)

def dim_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.reshape(combined_shape)

def complexpwr(a, b=None):
    if b is None:
        if torch.is_complex(a):
            real = a.real
            imag = a.imag
        real, imag = torch.split(a, [1, 1], dim=-1)
    else:
        real, imag = a, b
    data = (real ** 2 + imag ** 2)
    return data.squeeze(dim=-1)


def ludspkersimu(data):
    data = torch.clamp(data, max=0.8, min=-0.8)
    b = 1.5 * data - 0.3 * data ** 2
    a = (b>0).float()*3.5+0.5
    data = 4 * (2 / (1 + torch.exp(a * b)) - 1)
    return data


def single_roomconv(input, filter):
    if filter.shape[0] > input.shape[0]:
        filter = filter[0:input.shape[0]]
    if filter.shape[0] % 2 == 0:
        filter = filter[0:-1]
    out = F.conv1d(
        input.view([1, 1, -1]), filter.view([1, 1, -1]),
        padding=filter.shape[0] // 2).view([-1])
    return out


def multi_room_conv(input, filter):
    if filter.shape[-1] > input.shape[-1]:
        filter = filter[0:input.shape[-1]]
    if filter.shape[-1] % 2 == 0:
        filter = filter[:, 0:-1]
    out = F.conv1d(
        input.unsqueeze(0), filter.unsqueeze(1),
        padding=filter.shape[-1] // 2, groups=input.shape[0])
    return out.squeeze(dim=0)


def single_spk_and_echo(refdata, fltdata):
    echodata = single_roomconv(ludspkersimu(refdata), fltdata)
    return echodata


def multi_spk_and_echo(refdata, fltdata):
    echodata = multi_room_conv(ludspkersimu(refdata), fltdata)
    return echodata

def complex_spec_gain(src, gain):
    b, t, f, c = src.shape
    src = torch.mm(
        torch.diag_embed(gain), src.reshape(b, -1)).reshape(b, t, f, c)
    return src


def batch_spec_gain(src, gain):
    b, t, f = src.shape
    src = torch.mm(
        torch.diag_embed(gain), src.reshape(b, -1)).reshape(b, t, f)
    return src

def single_spec_gain(src, gain):
    t, f = src.shape
    src = torch.mm(
        torch.diag_embed(gain), src.reshape(b, -1)).reshape(t, f)
    return src