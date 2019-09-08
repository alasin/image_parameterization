import torch
import torch.nn as nn
from .common import *
import torch.nn.functional as F
from .common_func import *
from sklearn.random_projection import SparseRandomProjection as SRP
from scipy.sparse import find


class SkipIntrinsic(nn.Module):
    def __init__(self, num_input_channels=32, num_output_channels=3, 
                num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
                filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                need_sigmoid=True, need_bias=True, 
                pad='zero', upsample_mode='bilinear', downsample_mode='stride', act_fun='LeakyReLU', 
                need1x1_up=True, d=1000):
        super(SkipIntrinsic, self).__init__()

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
        self.num_parameters = self.count_parameters()
        self.d = d
        # self.transforms, self.projection_matrix = self.sample_transform_matrices_big(self.num_parameters, self.d)
        self.transforms, self.norm = self.sample_sparse_individual_transforms(self.num_parameters, self.d)
        
        
        print(self.num_parameters)
        self.make_untrainable()
        print(self.count_parameters())


        # self.subspace = torch.zeros((d, 1), requires_grad=True, device="cuda")
        # self.subspace = torch.div(self.subspace, self.norm)
        # self.subspace = torch.zeros((self.d, 1), dtype=torch.float, requires_grad=True)
        # print(self.count_parameters())


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_untrainable(self):
        for p in self.parameters():
            if p.requires_grad:
                p.requires_grad = False

    def sample_individual_transforms(self, D, d):
        mt = {}
        summation = torch.from_numpy(np.zeros((1, d), dtype=np.float32)).cuda()
        for i, p in enumerate(self.named_parameters()):
            num = 1
            name = p[0]
            layer_name = name.split('.')[0]
            if layer_name not in mt:
                mt[layer_name] = {}

            for k in p[1].shape:
                num = num*k; 
            
            matrix = torch.randn((num, d), dtype=torch.float, requires_grad=False).cuda()
            summation += torch.sum(torch.pow(matrix, 2), dim=0, keepdim=True)

            if 'weight' in name:  
                mt[layer_name]['weight'] = matrix
                mt[layer_name]['w_shape'] = p[1].shape
            elif 'bias' in name:
                mt[layer_name]['bias'] = matrix
                mt[layer_name]['b_shape'] = p[1].shape
            else:
                continue
        
        # print(summation)
        norm = torch.sqrt(summation)
        for i, p in enumerate(self.named_parameters()):
            name = p[0]
            layer_name = name.split('.')[0]
            if 'weight' in name:  
                mt[layer_name]['weight'] = torch.div(mt[layer_name]['weight'], norm)
                # print(mt[layer_name]['weight'])
            if 'bias' in name:
                mt[layer_name]['bias'] = torch.div(mt[layer_name]['bias'], norm)
        
        return mt

    def sample_sparse_individual_transforms(self, D, d):
        mt = {}
        summation = torch.from_numpy(np.zeros((d, 1), dtype=np.float32)).cuda()
        for i, p in enumerate(self.named_parameters()):
            num = 1
            name = p[0]
            layer_name = name.split('.')[0]
            if layer_name not in mt:
                mt[layer_name] = {}

            for k in p[1].shape:
                num = num*k; 
            
            M = SRP(d)._make_random_matrix(d, num)
            fm = find(M)

            # Create sparse projection matrix from small vv to full theta space
            matrix_sparse = torch.sparse.FloatTensor(indices=torch.LongTensor([fm[0], fm[1]]), values=torch.FloatTensor(fm[2]), size=torch.Size([d, num])).cuda()
            summation += torch.unsqueeze(torch.sparse.sum(torch.pow(matrix_sparse, 2), dim=1).to_dense(), -1)
            matrix_sparse.requires_grad = False
            
            if 'weight' in name:  
                mt[layer_name]['weight'] = matrix_sparse.coalesce().transpose(0, 1)
                mt[layer_name]['w_shape'] = p[1].shape
                mt[layer_name]['w_num'] = num
            elif 'bias' in name:
                mt[layer_name]['bias'] = matrix_sparse.coalesce().transpose(0, 1)
                mt[layer_name]['b_shape'] = p[1].shape
                mt[layer_name]['b_num'] = num
            else:
                continue
            
        # print(summation)
        norm = torch.sqrt(summation)
            
        return mt, norm


    def sample_transform_matrices(self, D, d):
        """Matrix P in the paper of size (D, d)
        Columns of P are normalized to unit length.
        """

        matrix = torch.randn((D, d), dtype=torch.float, requires_grad=False)
        # print(torch.sum(torch.pow(matrix, 2), dim=0, keepdim=True))
        y = torch.norm(matrix, dim=0, keepdim=True)
        matrix = torch.div(matrix, y)
        # split P according to num. params per layer.
        mt = {}
        offset = 0
        for i, p in enumerate(self.named_parameters()):
            num = 1
            name = p[0]
            layer_name = name.split('.')[0]
            if layer_name not in mt:
                mt[layer_name] = {}

            for k in p[1].shape:
                num = num*k; 
                
            w_matrix = matrix[offset:offset + num].cuda()
            # w_matrix = torch.from_numpy(w_matrix_values).type(torch.cuda.FloatTensor)
            if 'weight' in name:  
                mt[layer_name]['weight'] = w_matrix
                mt[layer_name]['w_shape'] = p[1].shape
            elif 'bias' in name:
                mt[layer_name]['bias'] = w_matrix
                mt[layer_name]['b_shape'] = p[1].shape
            else:
                continue            
            
            offset += num
                
        return mt
    
    def sample_transform_matrices_big(self, D, d):
        """Matrix P in the paper of size (D, d)
        Columns of P are normalized to unit length.
        """

        matrix = torch.randn((D, d), dtype=torch.float, requires_grad=False)
        y = torch.norm(matrix, dim=0, keepdim=True)
        matrix = torch.div(matrix, y).cuda()
        # split P according to num. params per layer.
        mt = {}
        offset = 0
        for i, p in enumerate(self.named_parameters()):
            num = 1
            name = p[0]
            layer_name = name.split('.')[0]
            if layer_name not in mt:
                mt[layer_name] = {}

            for k in p[1].shape:
                num = num*k; 
                
            # w_matrix = matrix[offset:offset + num].cuda()
            # w_matrix = torch.from_numpy(w_matrix_values).type(torch.cuda.FloatTensor)
            if 'weight' in name:
                mt[layer_name]['weight'] = None
                mt[layer_name]['w_offset'] = offset
                mt[layer_name]['w_num'] = num
                mt[layer_name]['w_shape'] = p[1].shape
            elif 'bias' in name:
                mt[layer_name]['bias'] = None
                mt[layer_name]['b_num'] = num
                mt[layer_name]['b_offset'] = offset
                mt[layer_name]['b_shape'] = p[1].shape
            else:
                continue            
            
            offset += num
                
        return mt, matrix

    
    def forward_new(self, x_orig, subspace):
        ##############################33#
        # print(self.projection_matrix.shape, subspace.shape)
        new_m = torch.matmul(self.projection_matrix, subspace)
        # print(new_m.shape)

        for i, p in enumerate(self.named_parameters()):
            name = p[0]
            layer_name = name.split('.')[0]
            # print(layer_name)
            layer = self.transforms[layer_name]
            w_offset = layer['w_offset']
            w_num = layer['w_num']
            b_offset = layer['b_offset']
            b_num = layer['b_num']
            if 'weight' in name:
                # print(w_offset, w_num)
                self.transforms[layer_name]['weight'] = new_m[w_offset:w_offset + w_num]
            if 'bias' in name:
                # print(b_offset, b_num)
                self.transforms[layer_name]['bias'] = new_m[b_offset:b_offset + b_num]

        
        layer = self.transforms['conv1_1']
        x = conv_f(x_orig, stride=2, kernel_size=3, layer=layer, subspace=subspace)
        # return x
        
        layer = self.transforms['bn1_1']
        x = bn_f(x, layer, self.bn1_1, subspace)
        x = act_f(x)

        layer = self.transforms['conv1_2']
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)
        
        layer = self.transforms['bn1_2']
        x = bn_f(x, layer, self.bn1_2, subspace)
        x1_out = act_f(x)
        ##################################

        ####################################
        layer = self.transforms['skip_conv1']
        x1_skip = conv_f(x_orig, stride=1, kernel_size=1, layer=layer, subspace=subspace)

        layer = self.transforms['skip_bn1']
        x1_skip = bn_f(x1_skip, layer, self.skip_bn1, subspace)
        x1_skip = act_f(x1_skip)
        ####################################

        ##############################33#
        layer = self.transforms['conv2_1']
        x = conv_f(x1_out, stride=2, kernel_size=3, layer=layer, subspace=subspace)
        
        layer = self.transforms['bn2_1']
        x = bn_f(x, layer, self.bn2_1, subspace)
        x = act_f(x)

        layer = self.transforms['conv2_2']
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)
        
        layer = self.transforms['bn2_2']
        x = bn_f(x, layer, self.bn2_2, subspace)
        x2_out = act_f(x)
        ##################################

        ####################################
        layer = self.transforms['skip_conv2']
        x2_skip = conv_f(x1_out, stride=1, kernel_size=1, layer=layer, subspace=subspace)

        layer = self.transforms['skip_bn2']
        x2_skip = bn_f(x2_skip, layer, self.skip_bn2, subspace)
        x2_skip = act_f(x2_skip)
        ####################################

        ##############################33#
        layer = self.transforms['conv3_1']  #1
        x = conv_f(x2_out, stride=2, kernel_size=3, layer=layer, subspace=subspace) #1
        
        layer = self.transforms['bn3_1']    #1
        x = bn_f(x, layer, self.bn3_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['conv3_2']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)
        
        layer = self.transforms['bn3_2']    #1
        x = bn_f(x, layer, self.bn3_2, subspace)  #1
        x3_out = act_f(x)   #1
        ##################################

        ####################################
        layer = self.transforms['skip_conv3']   #1
        x3_skip = conv_f(x2_out, stride=1, kernel_size=1, layer=layer, subspace=subspace)   #2

        layer = self.transforms['skip_bn3']  #1
        x3_skip = bn_f(x3_skip, layer, self.skip_bn3, subspace)   #3
        x3_skip = act_f(x3_skip)    #2
        ####################################

        ##############################33#
        layer = self.transforms['conv4_1']  #1
        x = conv_f(x3_out, stride=2, kernel_size=3, layer=layer, subspace=subspace) #1
        
        layer = self.transforms['bn4_1']    #1
        x = bn_f(x, layer, self.bn4_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['conv4_2']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)
        
        layer = self.transforms['bn4_2']    #1
        x = bn_f(x, layer, self.bn4_2, subspace)  #1
        x4_out = act_f(x)   #1
        ##################################

        ####################################
        layer = self.transforms['skip_conv4']   #1
        x4_skip = conv_f(x3_out, stride=1, kernel_size=1, layer=layer, subspace=subspace)   #2

        layer = self.transforms['skip_bn4']  #1
        x4_skip = bn_f(x4_skip, layer, self.skip_bn4, subspace)   #3
        x4_skip = act_f(x4_skip)    #2
        ####################################

        ##############################33#
        layer = self.transforms['conv5_1']  #1
        x = conv_f(x4_out, stride=2, kernel_size=3, layer=layer, subspace=subspace) #1
        
        layer = self.transforms['bn5_1']    #1
        x = bn_f(x, layer, self.bn5_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['conv5_2']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)
        
        layer = self.transforms['bn5_2']    #1
        x = bn_f(x, layer, self.bn5_2, subspace)  #1
        x5_out = act_f(x)   #1
        ##################################

        ####################################
        layer = self.transforms['skip_conv5']   #1
        x5_skip = conv_f(x4_out, stride=1, kernel_size=1, layer=layer, subspace=subspace)   #2

        layer = self.transforms['skip_bn5']  #1
        x5_skip = bn_f(x5_skip, layer, self.skip_bn5, subspace)   #3
        x5_skip = act_f(x5_skip)    #2
        ####################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x5_skip, x], dim=1)  #1
        
        layer = self.transforms['merge_bn5']  #1
        x = bn_f(x, layer, self.merge_bn5, subspace)   #1

        layer = self.transforms['dec_conv5_1']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)
        layer = self.transforms['dec_bn5_1']    #1
        x = bn_f(x, layer, self.dec_bn5_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['dec_conv5_2']  #1
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn5_2']    #1
        x = bn_f(x, layer, self.dec_bn5_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x4_skip, x], dim=1)
        
        layer = self.transforms['merge_bn4']  #1
        x = bn_f(x, layer, self.merge_bn4, subspace)   #1

        layer = self.transforms['dec_conv4_1']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn4_1']    #1
        
        x = bn_f(x, layer, self.dec_bn4_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['dec_conv4_2']  #1
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, subspace=subspace)
        # print(x.shape)
        layer = self.transforms['dec_bn4_2']    #1
        x = bn_f(x, layer, self.dec_bn4_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x3_skip, x], dim=1)
        # print(x.shape)
        
        layer = self.transforms['merge_bn3']  #1
        x = bn_f(x, layer, self.merge_bn3, subspace)   #1

        layer = self.transforms['dec_conv3_1']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn3_1']    #1
        x = bn_f(x, layer, self.dec_bn3_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['dec_conv3_2']  #1
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn3_2']    #1
        x = bn_f(x, layer, self.dec_bn3_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x2_skip, x], dim=1)
        
        layer = self.transforms['merge_bn2']  #1
        x = bn_f(x, layer, self.merge_bn2, subspace)   #1

        layer = self.transforms['dec_conv2_1']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn2_1']    #1
        x = bn_f(x, layer, self.dec_bn2_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['dec_conv2_2']  #1
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn2_2']    #1
        x = bn_f(x, layer, self.dec_bn2_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x1_skip, x], dim=1)  #1
        
        layer = self.transforms['merge_bn1']  #1
        x = bn_f(x, layer, self.merge_bn1, subspace)   #1

        layer = self.transforms['dec_conv1_1']  #1
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn1_1']    #1
        x = bn_f(x, layer, self.dec_bn1_1, subspace)  #1
        x = act_f(x)

        layer = self.transforms['dec_conv1_2']  #1
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, subspace=subspace)

        layer = self.transforms['dec_bn1_2']    #1
        x = bn_f(x, layer, self.dec_bn1_2, subspace)  #1
        x = act_f(x)
        #########################################################

        layer = self.transforms['final_conv']
        output = conv_f(x, stride=1, kernel_size=1, layer=layer, subspace=subspace)
        output = F.sigmoid(output)
        
        return output
    
    def forward(self, x_orig, subspace):
        subspace = torch.div(subspace, self.norm)
        # subspace = torch.floor(subspace)
        ##############################33#
        transform = self.transforms['conv1_1']
        layer = self.conv1_1[1]

        x = conv_f(x_orig, stride=2, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        # return x
        
        transform = self.transforms['bn1_1']
        x = bn_f(x, transform=transform, bn_module=self.bn1_1, subspace=subspace)
        x = act_f(x)

        transform = self.transforms['conv1_2']
        layer = self.conv1_2[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['bn1_2']
        x = bn_f(x, transform, self.bn1_2, subspace)
        x1_out = act_f(x)
        ##################################

        ####################################
        transform = self.transforms['skip_conv1']
        layer = self.skip_conv1[1]
        x1_skip = conv_f(x_orig, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['skip_bn1']
        x1_skip = bn_f(x1_skip, transform, self.skip_bn1, subspace)
        x1_skip = act_f(x1_skip)
        ####################################

        ##############################33#
        transform = self.transforms['conv2_1']
        layer = self.conv2_1[1]
        x = conv_f(x1_out, stride=2, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['bn2_1']
        x = bn_f(x, transform, self.bn2_1, subspace)
        x = act_f(x)

        transform = self.transforms['conv2_2']
        layer = self.conv2_2[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['bn2_2']
        x = bn_f(x, transform, self.bn2_2, subspace)
        x2_out = act_f(x)
        ##################################

        ####################################
        transform = self.transforms['skip_conv2']
        layer = self.skip_conv2[1] 
        x2_skip = conv_f(x1_out, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['skip_bn2']
        x2_skip = bn_f(x2_skip, transform, self.skip_bn2, subspace)
        x2_skip = act_f(x2_skip)
        ####################################

        ##############################33#
        transform = self.transforms['conv3_1']  #1
        layer = self.conv3_1[1]
        x = conv_f(x2_out, stride=2, kernel_size=3, layer=layer, transform=transform, subspace=subspace) #1
        
        transform = self.transforms['bn3_1']    #1
        x = bn_f(x, transform, self.bn3_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['conv3_2']  #1
        layer = self.conv3_2[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['bn3_2']    #1
        x = bn_f(x, transform, self.bn3_2, subspace)  #1
        x3_out = act_f(x)   #1
        ##################################

        ####################################
        transform = self.transforms['skip_conv3']   #1
        layer = self.skip_conv3[1]
        x3_skip = conv_f(x2_out, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)   #2

        transform = self.transforms['skip_bn3']  #1
        x3_skip = bn_f(x3_skip, transform, self.skip_bn3, subspace)   #3
        x3_skip = act_f(x3_skip)    #2
        ####################################

        ##############################33#
        transform = self.transforms['conv4_1']  #1
        layer = self.conv4_1[1]
        x = conv_f(x3_out, stride=2, kernel_size=3, layer=layer, transform=transform, subspace=subspace) #1
        
        transform = self.transforms['bn4_1']    #1
        x = bn_f(x, transform, self.bn4_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['conv4_2']  #1
        layer = self.conv4_2[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['bn4_2']    #1
        x = bn_f(x, transform, self.bn4_2, subspace)  #1
        x4_out = act_f(x)   #1
        ##################################

        ####################################
        transform = self.transforms['skip_conv4']   #1
        layer = self.skip_conv4[1]
        x4_skip = conv_f(x3_out, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)   #2

        transform = self.transforms['skip_bn4']  #1
        x4_skip = bn_f(x4_skip, transform, self.skip_bn4, subspace)   #3
        x4_skip = act_f(x4_skip)    #2
        ####################################

        ##############################33#
        transform = self.transforms['conv5_1']  #1
        layer = self.conv5_1[1]
        x = conv_f(x4_out, stride=2, kernel_size=3, layer=layer, transform=transform, subspace=subspace) #1
        
        transform = self.transforms['bn5_1']    #1
        x = bn_f(x, transform, self.bn5_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['conv5_2']  #1
        layer = self.conv5_2[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['bn5_2']    #1
        x = bn_f(x, transform, self.bn5_2, subspace)  #1
        x5_out = act_f(x)   #1
        ##################################

        ####################################
        transform = self.transforms['skip_conv5']   #1
        layer = self.skip_conv5[1]
        x5_skip = conv_f(x4_out, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)   #2

        transform = self.transforms['skip_bn5']  #1
        x5_skip = bn_f(x5_skip, transform, self.skip_bn5, subspace)   #3
        x5_skip = act_f(x5_skip)    #2
        ####################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x5_skip, x], dim=1)  #1
        
        transform = self.transforms['merge_bn5']  #1
        x = bn_f(x, transform, self.merge_bn5, subspace)   #1

        transform = self.transforms['dec_conv5_1']  #1
        layer = self.dec_conv5_1[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)
        
        transform = self.transforms['dec_bn5_1']    #1
        x = bn_f(x, transform, self.dec_bn5_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['dec_conv5_2']  #1
        layer = self.dec_conv5_2[1]
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn5_2']    #1
        x = bn_f(x, transform, self.dec_bn5_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x4_skip, x], dim=1)
        
        transform = self.transforms['merge_bn4']  #1
        x = bn_f(x, transform, self.merge_bn4, subspace)   #1

        transform = self.transforms['dec_conv4_1']  #1
        layer = self.dec_conv4_1[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn4_1']    #1
        
        x = bn_f(x, transform, self.dec_bn4_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['dec_conv4_2']  #1
        layer = self.dec_conv4_2[1]
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)
        # print(x.shape)
        transform = self.transforms['dec_bn4_2']    #1
        x = bn_f(x, transform, self.dec_bn4_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x3_skip, x], dim=1)
        # print(x.shape)
        
        transform = self.transforms['merge_bn3']  #1
        x = bn_f(x, transform, self.merge_bn3, subspace)   #1

        transform = self.transforms['dec_conv3_1']  #1
        layer = self.dec_conv3_1[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn3_1']    #1
        x = bn_f(x, transform, self.dec_bn3_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['dec_conv3_2']  #1
        layer = self.dec_conv3_2[1]
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn3_2']    #1
        x = bn_f(x, transform, self.dec_bn3_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x2_skip, x], dim=1)
        
        transform = self.transforms['merge_bn2']  #1
        x = bn_f(x, transform, self.merge_bn2, subspace)   #1

        transform = self.transforms['dec_conv2_1']  #1
        layer = self.dec_conv2_1[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn2_1']    #1
        x = bn_f(x, transform, self.dec_bn2_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['dec_conv2_2']  #1
        layer = self.dec_conv2_2[1]
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn2_2']    #1
        x = bn_f(x, transform, self.dec_bn2_2, subspace)  #1
        x = act_f(x)
        #########################################################

        #########################################################
        x = F.upsample(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat([x1_skip, x], dim=1)  #1
        
        transform = self.transforms['merge_bn1']  #1
        x = bn_f(x, transform, self.merge_bn1, subspace)   #1

        transform = self.transforms['dec_conv1_1']  #1
        layer = self.dec_conv1_1[1]
        x = conv_f(x, stride=1, kernel_size=3, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn1_1']    #1
        x = bn_f(x, transform, self.dec_bn1_1, subspace)  #1
        x = act_f(x)

        transform = self.transforms['dec_conv1_2']  #1
        layer = self.dec_conv1_2[1]
        x = conv_f(x, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)

        transform = self.transforms['dec_bn1_2']    #1
        x = bn_f(x, transform, self.dec_bn1_2, subspace)  #1
        x = act_f(x)
        #########################################################

        transform = self.transforms['final_conv']
        layer = self.final_conv[1]
        output = conv_f(x, stride=1, kernel_size=1, layer=layer, transform=transform, subspace=subspace)
        # print(torch.min(output), torch.max(output))
        # sys.exit()
        output = F.sigmoid(output)
        
        return output