import torch
import torch.nn as nn
import copy
from ptflops import get_model_complexity_info
from collections import OrderedDict

device = 'cuda:0'
 
def fcConvWeightReguViaSVB(model, prune_index_list):
    layer_count = 0
    
    for name, m in model.named_modules():   
        if isinstance(m,nn.Linear):
        # if isinstance(m,nn.Linear) and 'fn.net' in name: 
        # if isinstance(m,nn.Linear) and 'mlp' in name:
            prun_index = prune_index_list[layer_count]
            layer_count += 1


        
            tmpbatchM = m.weight.data.t().clone()

            tmpU, tmpS, tmpV = torch.linalg.svd(tmpbatchM, full_matrices=False)


            
            alpha = ( torch.norm(tmpS, p=2).pow(2) - torch.norm((tmpS[prun_index:] * 0), p=2).pow(2) ) / torch.norm(tmpS[:prun_index], p=2).pow(2)
            tmpS[:prun_index] = tmpS[:prun_index] * torch.sqrt(alpha)  
            tmpS[prun_index:] = tmpS[prun_index:] * 0
            tmpbatchMx = torch.mm(torch.mm(tmpU, torch.diag(tmpS.to(device))), tmpV).contiguous()
            tmpbatchMx = tmpbatchMx.t()
            m.weight.data.copy_(tmpbatchMx.view_as(m.weight.data))

    return model 
 
def _set_model_attr(field_name, att, obj):
    '''
    set a certain filed_name like 'xx.xx.xx' as the object att
    :param field_name: str, 'xx.xx.xx' to indicate the attribution tree
    :param att: input object to replace a certain field attribution
    :param obj: objective attribution
    '''

    field_list = field_name.split('.')
    a = att

    # to achieve the second last level of attr tree
    for field in field_list[:-1]:
        a = getattr(a, field)

    setattr(a, field_list[-1], obj)

 
def channel_decompose_guss(model_in, prune_index_list):
    '''
    decouple a input pre-trained model under nuclear regularization
    with singular value decomposition
    a single NxCxHxW low-rank filter is decoupled
    into a NxRx1x1 kernel following a RxCxHxW kernel
    :param model_in: object of derivated class of nn.Module, the model is initialized with pre-trained weight
    :param look_up_table: list, containing module names to be decouple
    :param criterion: object, a filter to filter out small valued simgas, only valid when train is False
    :param train: bool, whether decompose during training, if true, function only compute corresponding
           gradient w.r.t each singular value and do not apply actual decouple
    :param lambda_: float, weight for regularization term, only valid when train is True
    :return: model_out: a new nn.Module object initialized with a decoupled model
    '''
    layer_count = 0
    for name, m in model_in.named_modules():
        prun_flag = False
        if isinstance(m,nn.Linear):
        # if isinstance(m,nn.Linear) and 'fn.net' in name:
        # if isinstance(m,nn.Linear) and 'mlp' in name:
            prun_index = prune_index_list[layer_count]
            layer_count += 1
            # if name in look_up_table:
            param = m.weight.data
            dim = param.size()
            
            if m.bias is not None:             
                hasb = True
                b = m.bias.data
            else:
                hasb = False
            
            NC = param.view(dim[0], -1) # [N x CHW]

            # try:
            N, sigma, C = torch.svd(NC, some=True)
            C = C.t()

            # prune
            if sigma[0] < 1e-5:
                prun_flag = True

            N = N[:, :prun_index].contiguous()
            sigma = sigma[:prun_index]
            C = C[:prun_index, :]

            
            r = int(sigma.size(0))
            C = torch.mm(torch.diag(torch.sqrt(sigma)), C)
            N = torch.mm(N,torch.diag(torch.sqrt(sigma)))

            C = C.view(r,dim[1])
            N = N.view(dim[0], r)

            if prun_flag == True:
                new_m = nn.Sequential()
            else:
                new_m = nn.Sequential(
                    OrderedDict([
                        ('C', nn.Linear(dim[1], r, bias=False)),
                        ('N', nn.Linear(r, dim[0], bias=hasb))
                    ])
                )
    
            
                state_dict = new_m.state_dict()
                print(name+'.C.weight'+' <-- '+name+'.weight')
                state_dict['C.weight'].copy_(C)
                print(name + '.N.weight' + ' <-- ' + name + '.weight')

                state_dict['N.weight'].copy_(N)
                if hasb:
                    print(name+'.N.bias'+' <-- '+name+'.bias')
                    state_dict['N.bias'].copy_(b)

                new_m.load_state_dict(state_dict)
            _set_model_attr(name, att=model_in, obj=new_m)


    return model_in.to(device)

def ratio_print(model,prune_index_list):
    model_test = copy.deepcopy(model)

    flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=True)

    model_pruning = channel_decompose_guss(model_in=model_test, prune_index_list=prune_index_list)

    flops_svd, params_svd = get_model_complexity_info(model_pruning, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    print(params, '  ', params_svd)
    print(flops, '   ', flops_svd)
    print('pruning ratio param：', (1-(params_svd / params)))
    print('pruning ratio flops：', (1-(flops_svd / flops)))