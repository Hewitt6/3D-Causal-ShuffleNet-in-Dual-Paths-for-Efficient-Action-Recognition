import torch
from torch import nn

from models import shufflenet, causalshuf, causalshuf_dep, twowayshuf

def generate_model(opt):
    assert opt.model in ['shufflenet', 'causalshuf', 'slowfastshuf', 'slowfastcausalshuf', 'causalshuf_dep', 'twowayshuf']


    
    if opt.model in ['shufflenet', 'slowfastshuf']:
        from models.shufflenet import get_fine_tuning_parameters
        model = shufflenet.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)
    
    if opt.model in ['causalshuf', 'slowfastcausalshuf']:
        from models.causalshuf import get_fine_tuning_parameters
        model = causalshuf.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)

    if opt.model == 'causalshuf_dep':
        from models.causalshuf_dep import get_fine_tuning_parameters
        model = causalshuf_dep.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)

    if opt.model == 'twowayshuf':
        from models.twowayshuf import get_fine_tuning_parameters
        model = twowayshuf.get_model(
            groups=opt.groups,
            width_mult=opt.width_mult,
            num_classes=opt.n_classes)

    
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['shufflenet', 'causalshuf', 'slowfastshuf', 'slowfastcausalshuf']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_portion)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['shufflenet', 'causalshuf', 'slowfastshuf', 'slowfastcausalshuf']:
                model.module.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.module.classifier[1].in_features, opt.n_finetune_classes)
                                )
            

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
