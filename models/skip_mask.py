import torch
import torch.nn as nn
from .common import *
import torch.nn.functional as F


class SkipMaskGen(nn.Module):
    def __init__(self, num_input_channels=32, num_output_channels=3, 
                num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
                filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                need_sigmoid=True, need_bias=True, 
                pad='reflection', upsample_mode='bilinear', downsample_mode='stride', act_fun='ReLU', 
                need1x1_up=True):
        super(SkipMaskGen, self).__init__()

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down)

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
        
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            filter_size_down   = [filter_size_down]*n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            filter_size_up   = [filter_size_up]*n_scales

        self.upsample_mode = upsample_mode

        # model = nn.Sequential()
        # model_tmp = model

        input_depth = num_input_channels

        self.act = act(act_fun)

        self.conv1_1 = conv(input_depth, num_channels_down[0], filter_size_down[0], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[0])
        self.bn1_1 = bn(num_channels_down[0])

        self.conv1_2 = conv(num_channels_down[0], num_channels_down[0], filter_size_down[0], bias=need_bias, pad=pad)
        self.bn1_2 = bn(num_channels_down[0])

        self.skip_conv1 = conv(input_depth, num_channels_skip[0], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn1 = bn(num_channels_skip[0])

        input_depth = num_channels_down[0]
        self.conv2_1 = conv(input_depth, num_channels_down[1], filter_size_down[1], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[1])
        self.bn2_1 = bn(num_channels_down[1])

        self.conv2_2 = conv(num_channels_down[1], num_channels_down[1], filter_size_down[1], bias=need_bias, pad=pad)
        self.bn2_2 = bn(num_channels_down[1])

        self.skip_conv2 = conv(input_depth, num_channels_skip[1], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn2 = bn(num_channels_skip[1])

        input_depth = num_channels_down[1]
        self.conv3_1 = conv(input_depth, num_channels_down[2], filter_size_down[2], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[2])
        self.bn3_1 = bn(num_channels_down[2])

        self.conv3_2 = conv(num_channels_down[2], num_channels_down[2], filter_size_down[2], bias=need_bias, pad=pad)
        self.bn3_2 = bn(num_channels_down[2])

        self.skip_conv3 = conv(input_depth, num_channels_skip[2], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn3 = bn(num_channels_skip[2])

        input_depth = num_channels_down[2]
        self.conv4_1 = conv(input_depth, num_channels_down[3], filter_size_down[3], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[3])
        self.bn4_1 = bn(num_channels_down[3])

        self.conv4_2 = conv(num_channels_down[3], num_channels_down[3], filter_size_down[3], bias=need_bias, pad=pad)
        self.bn4_2 = bn(num_channels_down[3])

        self.skip_conv4 = conv(input_depth, num_channels_skip[3], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn4 = bn(num_channels_skip[3])

        input_depth = num_channels_down[3]
        self.conv5_1 = conv(input_depth, num_channels_down[4], filter_size_down[4], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[4])
        self.bn5_1 = bn(num_channels_down[4])

        self.conv5_2 = conv(num_channels_down[4], num_channels_down[4], filter_size_down[4], bias=need_bias, pad=pad)
        self.bn5_2 = bn(num_channels_down[4])

        self.skip_conv5 = conv(input_depth, num_channels_skip[4], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn5 = bn(num_channels_skip[4])
        
        k = num_channels_up[1]
        self.dec_conv1_1 = conv(num_channels_skip[0] + k, num_channels_up[0], filter_size_up[0], 1, bias=need_bias, pad=pad)
        self.dec_bn1_1 = bn(num_channels_up[0])

        self.dec_conv1_2 = conv(num_channels_up[0], num_channels_up[0], 1, bias=need_bias, pad=pad)
        self.dec_bn1_2 = bn(num_channels_up[0])

        k = num_channels_up[2]
        self.dec_conv2_1 = conv(num_channels_skip[1] + k, num_channels_up[1], filter_size_up[1], 1, bias=need_bias, pad=pad)
        self.dec_bn2_1 = bn(num_channels_up[1])

        self.dec_conv2_2 = conv(num_channels_up[1], num_channels_up[1], 1, bias=need_bias, pad=pad)
        self.dec_bn2_2 = bn(num_channels_up[1])

        k = num_channels_up[3]
        self.dec_conv3_1 = conv(num_channels_skip[2] + k, num_channels_up[2], filter_size_up[2], 1, bias=need_bias, pad=pad)
        self.dec_bn3_1 = bn(num_channels_up[2])

        self.dec_conv3_2 = conv(num_channels_up[2], num_channels_up[2], 1, bias=need_bias, pad=pad)
        self.dec_bn3_2 = bn(num_channels_up[2])

        k = num_channels_up[4]
        self.dec_conv4_1 = conv(num_channels_skip[3] + k, num_channels_up[3], filter_size_up[3], 1, bias=need_bias, pad=pad)
        self.dec_bn4_1 = bn(num_channels_up[3])

        self.dec_conv4_2 = conv(num_channels_up[3], num_channels_up[3], 1, bias=need_bias, pad=pad)
        self.dec_bn4_2 = bn(num_channels_up[3])

        k = num_channels_down[4]
        self.dec_conv5_1 = conv(num_channels_skip[4] + k, num_channels_up[4], filter_size_up[4], 1, bias=need_bias, pad=pad)
        self.dec_bn5_1 = bn(num_channels_up[4])

        self.dec_conv5_2 = conv(num_channels_up[4], num_channels_up[4], 1, bias=need_bias, pad=pad)
        self.dec_bn5_2 = bn(num_channels_up[4])

        self.merge_bn5 = bn(num_channels_down[4] + num_channels_skip[4])
        self.merge_bn4 = bn(num_channels_skip[3] + (num_channels_up[4]))
        self.merge_bn3 = bn(num_channels_skip[2] + (num_channels_up[3]))
        self.merge_bn2 = bn(num_channels_skip[1] + (num_channels_up[2]))
        self.merge_bn1 = bn(num_channels_skip[0] + (num_channels_up[1]))

        self.final_conv = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)

    def forward(self, x_orig):
        mask_array = []
        x = self.conv1_1(x_orig)
        x = self.bn1_1(x)
        x = self.act(x)

        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x1_out = self.act(x)
        
        mask = x1_out > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x1_skip = self.skip_conv1(x_orig)
        x1_skip = self.skip_bn1(x1_skip)
        x1_skip = self.act(x1_skip)
        
        mask = x1_skip > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = self.conv2_1(x1_out)
        x = self.bn2_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x2_out = self.act(x)
        
        mask = x2_out > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x2_skip = self.skip_conv2(x1_out)
        x2_skip = self.skip_bn2(x2_skip)
        x2_skip = self.act(x2_skip)
        
        mask = x2_skip > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = self.conv3_1(x2_out)
        x = self.bn3_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x3_out = self.act(x)
        
        mask = x3_out > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x3_skip = self.skip_conv3(x2_out)
        x3_skip = self.skip_bn3(x3_skip)
        x3_skip = self.act(x3_skip)
        
        mask = x3_skip > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = self.conv4_1(x3_out)
        x = self.bn4_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x4_out = self.act(x)
        
        mask = x4_out > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x4_skip = self.skip_conv4(x3_out)
        x4_skip = self.skip_bn4(x4_skip)
        x4_skip = self.act(x4_skip)
        
        mask = x4_skip > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = self.conv5_1(x4_out)
        x = self.bn5_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x5_skip = self.skip_conv5(x4_out)
        x5_skip = self.skip_bn5(x5_skip)
        x5_skip = self.act(x5_skip)
        
        mask = x5_skip > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x5_skip, x], dim=1)
        x = self.merge_bn5(x)
        

        x = self.dec_conv5_1(x)
        x = self.dec_bn5_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.dec_conv5_2(x)
        x = self.dec_bn5_2(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x4_skip, x], dim=1)
        x = self.merge_bn4(x)
        

        x = self.dec_conv4_1(x)
        x = self.dec_bn4_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.dec_conv4_2(x)
        x = self.dec_bn4_2(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x3_skip, x], dim=1)
        x = self.merge_bn3(x)
        

        x = self.dec_conv3_1(x)
        x = self.dec_bn3_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.dec_conv3_2(x)
        x = self.dec_bn3_2(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x2_skip, x], dim=1)
        x = self.merge_bn2(x)
        

        x = self.dec_conv2_1(x)
        x = self.dec_bn2_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.dec_conv2_2(x)
        x = self.dec_bn2_2(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x1_skip, x], dim=1)
        x = self.merge_bn1(x)
        

        x = self.dec_conv1_1(x)
        x = self.dec_bn1_1(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())
        
        x = self.dec_conv1_2(x)
        x = self.dec_bn1_2(x)
        x = self.act(x)
        
        mask = x > 0
        mask.requires_grad = False
        mask_array.append(mask.detach())

        output = self.final_conv(x)
        output = F.sigmoid(output)
        
        return output, mask_array


class SkipMaskApply(nn.Module):
    def __init__(self, num_input_channels=32, num_output_channels=3, 
                num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
                filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                need_sigmoid=True, need_bias=True, 
                pad='reflection', upsample_mode='bilinear', downsample_mode='stride', act_fun='ReLU', 
                need1x1_up=True):
        super(SkipMaskApply, self).__init__()

        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down)

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
        
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            filter_size_down   = [filter_size_down]*n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            filter_size_up   = [filter_size_up]*n_scales

        self.upsample_mode = upsample_mode

        input_depth = num_input_channels

        self.conv1_1 = conv(input_depth, num_channels_down[0], filter_size_down[0], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[0])
        self.bn1_1 = bn(num_channels_down[0])
        # self.act1_1 = act(act_fun)

        self.conv1_2 = conv(num_channels_down[0], num_channels_down[0], filter_size_down[0], bias=need_bias, pad=pad)
        self.bn1_2 = bn(num_channels_down[0])
        # self.act1_2 = act(act_fun)

        self.skip_conv1 = conv(input_depth, num_channels_skip[0], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn1 = bn(num_channels_skip[0])
        # self.skip_act1 = act(act_fun)

        input_depth = num_channels_down[0]
        self.conv2_1 = conv(input_depth, num_channels_down[1], filter_size_down[1], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[1])
        self.bn2_1 = bn(num_channels_down[1])
        # self.act2_1 = act(act_fun)

        self.conv2_2 = conv(num_channels_down[1], num_channels_down[1], filter_size_down[1], bias=need_bias, pad=pad)
        self.bn2_2 = bn(num_channels_down[1])
        # self.act2_2 = act(act_fun)

        self.skip_conv2 = conv(input_depth, num_channels_skip[1], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn2 = bn(num_channels_skip[1])
        # self.skip_act2 = act(act_fun)

        input_depth = num_channels_down[1]
        self.conv3_1 = conv(input_depth, num_channels_down[2], filter_size_down[2], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[2])
        self.bn3_1 = bn(num_channels_down[2])
        # self.act3_1 = act(act_fun)

        self.conv3_2 = conv(num_channels_down[2], num_channels_down[2], filter_size_down[2], bias=need_bias, pad=pad)
        self.bn3_2 = bn(num_channels_down[2])
        # self.act3_2 = act(act_fun)

        self.skip_conv3 = conv(input_depth, num_channels_skip[2], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn3 = bn(num_channels_skip[2])
        # self.skip_act3 = act(act_fun)

        input_depth = num_channels_down[2]
        self.conv4_1 = conv(input_depth, num_channels_down[3], filter_size_down[3], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[3])
        self.bn4_1 = bn(num_channels_down[3])
        # self.act4_1 = act(act_fun)

        self.conv4_2 = conv(num_channels_down[3], num_channels_down[3], filter_size_down[3], bias=need_bias, pad=pad)
        self.bn4_2 = bn(num_channels_down[3])
        # self.act4_2 = act(act_fun)

        self.skip_conv4 = conv(input_depth, num_channels_skip[3], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn4 = bn(num_channels_skip[3])
        # self.skip_act4 = act(act_fun)

        input_depth = num_channels_down[3]
        self.conv5_1 = conv(input_depth, num_channels_down[4], filter_size_down[4], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[4])
        self.bn5_1 = bn(num_channels_down[4])
        # self.act5_1 = act(act_fun)

        self.conv5_2 = conv(num_channels_down[4], num_channels_down[4], filter_size_down[4], bias=need_bias, pad=pad)
        self.bn5_2 = bn(num_channels_down[4])
        # self.act5_2 = act(act_fun)

        self.skip_conv5 = conv(input_depth, num_channels_skip[4], filter_skip_size, bias=need_bias, pad=pad)
        self.skip_bn5 = bn(num_channels_skip[4])
        # self.skip_act5 = act(act_fun)
        
        k = num_channels_up[1]
        self.dec_conv1_1 = conv(num_channels_skip[0] + k, num_channels_up[0], filter_size_up[0], 1, bias=need_bias, pad=pad)
        self.dec_bn1_1 = bn(num_channels_up[0])
        # self.dec_act1_1 = act(act_fun)

        self.dec_conv1_2 = conv(num_channels_up[0], num_channels_up[0], 1, bias=need_bias, pad=pad)
        self.dec_bn1_2 = bn(num_channels_up[0])
        # self.dec_act1_2 = act(act_fun)

        k = num_channels_up[2]
        self.dec_conv2_1 = conv(num_channels_skip[1] + k, num_channels_up[1], filter_size_up[1], 1, bias=need_bias, pad=pad)
        self.dec_bn2_1 = bn(num_channels_up[1])
        # self.dec_act2_1 = act(act_fun)

        self.dec_conv2_2 = conv(num_channels_up[1], num_channels_up[1], 1, bias=need_bias, pad=pad)
        self.dec_bn2_2 = bn(num_channels_up[1])
        # self.dec_act2_2 = act(act_fun)

        k = num_channels_up[3]
        self.dec_conv3_1 = conv(num_channels_skip[2] + k, num_channels_up[2], filter_size_up[2], 1, bias=need_bias, pad=pad)
        self.dec_bn3_1 = bn(num_channels_up[2])
        # self.dec_act3_1 = act(act_fun)

        self.dec_conv3_2 = conv(num_channels_up[2], num_channels_up[2], 1, bias=need_bias, pad=pad)
        self.dec_bn3_2 = bn(num_channels_up[2])
        # self.dec_act3_2 = act(act_fun)

        k = num_channels_up[4]
        self.dec_conv4_1 = conv(num_channels_skip[3] + k, num_channels_up[3], filter_size_up[3], 1, bias=need_bias, pad=pad)
        self.dec_bn4_1 = bn(num_channels_up[3])
        # self.dec_act4_1 = act(act_fun)

        self.dec_conv4_2 = conv(num_channels_up[3], num_channels_up[3], 1, bias=need_bias, pad=pad)
        self.dec_bn4_2 = bn(num_channels_up[3])
        # self.dec_act4_2 = act(act_fun)

        k = num_channels_down[4]
        self.dec_conv5_1 = conv(num_channels_skip[4] + k, num_channels_up[4], filter_size_up[4], 1, bias=need_bias, pad=pad)
        self.dec_bn5_1 = bn(num_channels_up[4])
        # self.dec_act5_1 = act(act_fun)

        self.dec_conv5_2 = conv(num_channels_up[4], num_channels_up[4], 1, bias=need_bias, pad=pad)
        self.dec_bn5_2 = bn(num_channels_up[4])
        # self.dec_act5_2 = act(act_fun)

        self.merge_bn5 = bn(num_channels_down[4] + num_channels_skip[4])
        self.merge_bn4 = bn(num_channels_skip[3] + (num_channels_up[4]))
        self.merge_bn3 = bn(num_channels_skip[2] + (num_channels_up[3]))
        self.merge_bn2 = bn(num_channels_skip[1] + (num_channels_up[2]))
        self.merge_bn1 = bn(num_channels_skip[0] + (num_channels_up[1]))

        self.final_conv = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)

    def forward(self, x_orig, mask_array):
        i = -1
        x = self.conv1_1(x_orig)
        x = self.bn1_1(x)

        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)


        x = self.conv1_2(x)
        x = self.bn1_2(x)

        i += 1
        mask = mask_array[i].float()
        x1_out = torch.mul(x, mask)
        

        x1_skip = self.skip_conv1(x_orig)
        x1_skip = self.skip_bn1(x1_skip)
        
        i += 1
        mask = mask_array[i].float()
        x1_skip = torch.mul(x1_skip, mask)

        x = self.conv2_1(x1_out)
        x = self.bn2_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.conv2_2(x)
        x = self.bn2_2(x)

        i += 1
        mask = mask_array[i].float()
        x2_out = torch.mul(x, mask)

        x2_skip = self.skip_conv2(x1_out)
        x2_skip = self.skip_bn2(x2_skip)
        
        i += 1
        mask = mask_array[i].float()
        x2_skip = torch.mul(x2_skip, mask)

        x = self.conv3_1(x2_out)
        x = self.bn3_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x3_out = torch.mul(x, mask)
        

        x3_skip = self.skip_conv3(x2_out)
        x3_skip = self.skip_bn3(x3_skip)
        
        i += 1
        mask = mask_array[i].float()
        x3_skip = torch.mul(x3_skip, mask)

        x = self.conv4_1(x3_out)
        x = self.bn4_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x4_out = torch.mul(x, mask)

        x4_skip = self.skip_conv4(x3_out)
        x4_skip = self.skip_bn4(x4_skip)
        
        i += 1
        mask = mask_array[i].float()
        x4_skip = torch.mul(x4_skip, mask)

        x = self.conv5_1(x4_out)
        x = self.bn5_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        x5_skip = self.skip_conv5(x4_out)
        x5_skip = self.skip_bn5(x5_skip)
        
        i += 1
        mask = mask_array[i].float()
        x5_skip = torch.mul(x5_skip, mask)

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x5_skip, x], dim=1)
        x = self.merge_bn5(x)

        x = self.dec_conv5_1(x)
        x = self.dec_bn5_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.dec_conv5_2(x)
        x = self.dec_bn5_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x4_skip, x], dim=1)
        x = self.merge_bn4(x)

        x = self.dec_conv4_1(x)
        x = self.dec_bn4_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.dec_conv4_2(x)
        x = self.dec_bn4_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x3_skip, x], dim=1)
        x = self.merge_bn3(x)

        x = self.dec_conv3_1(x)
        x = self.dec_bn3_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.dec_conv3_2(x)
        x = self.dec_bn3_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x2_skip, x], dim=1)
        x = self.merge_bn2(x)
        

        x = self.dec_conv2_1(x)
        x = self.dec_bn2_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.dec_conv2_2(x)
        x = self.dec_bn2_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x1_skip, x], dim=1)
        x = self.merge_bn1(x)

        x = self.dec_conv1_1(x)
        x = self.dec_bn1_1(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)
        
        x = self.dec_conv1_2(x)
        x = self.dec_bn1_2(x)
        
        i += 1
        mask = mask_array[i].float()
        x = torch.mul(x, mask)

        output = self.final_conv(x)
        output = F.sigmoid(output)
        
        return output