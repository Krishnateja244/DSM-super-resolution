import os
import torch

MODEL_DIR = '/home/nall_kr/Documents/sr_dsm/D-SRGAN/LAM_Demo/ModelZoo/models'


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
    'SRGAN',
    'ESRGAN',
    'SRCAGAN',
    'enc_srgan'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'SRGAN': {
        'Base': 'model_best_srgan_nobn_self.pt'
    },
    'ESRGAN': {
        'Base' :'model_best_esrgan_patch.pt'
    },
    'SRCAGAN': {
        'Base' : 'model_best_srcagan_nobn.pt'
    },
    'enc_srgan': {
        'Base' : 'model_best_enc_decodernewM3.pt'
    }
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)
        
        elif model_name == "SRGAN":
            from .NN.generator import SrganGenerator
            net = SrganGenerator(1,128,4)

        elif model_name == "ESRGAN":
            from .NN.ESRGAN import ESRGAN 
            net = ESRGAN(1,1,scale_factor=4)

        elif model_name == "SRCAGAN":
            from .NN.srgan_resca import SRCAGAN
            net = SRCAGAN(1,128,10,10,16)

        elif model_name == "enc_srgan":
            from .NN.generator import Encoder
            net = Encoder(1)

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    print(state_dict_path)
    state_dict = torch.load(state_dict_path, map_location='cpu')
    net.load_state_dict(state_dict["g_model"])
    return net




