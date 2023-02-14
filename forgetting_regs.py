import enum
import torch
# from baseline_tools import IntermediateLayerGetter, calculate_channel_attention, GTOT, calculate_fishers, calculate_channel_attention_link_prediction
import os 
from torch import nn 
from collections import OrderedDict
import torch_geometric.utils as PyG_utils
import functools

def get_attribute(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class IntermediateLayerGetter:
    r"""
    Wraps a model to get intermediate output values of selected layers.

    Args:
       model (torch.nn.Module): The model to collect intermediate layer feature maps.
       return_layers (list): The names of selected modules to return the output.
       keep_output (bool): If True, `model_output` contains the final model's output, else return None. Default: True

    Returns:
       - An OrderedDict of intermediate outputs. The keys are selected layer names in `return_layers` and the values are the feature map outputs. The order is the same as `return_layers`.
       - The model's final output. If `keep_output` is False, return None.

    """
    def __init__(self, model, return_layers, keep_output=True):
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name in self.return_layers:
            layer = get_attribute(self._model, name)
            def hook(module, input, output, name=name):
                ret[name] = output
            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)

        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, output

def feature_regs(source_model, target_model, return_layers):
    source_getter = IntermediateLayerGetter(source_model, return_layers=return_layers)
    target_getter = IntermediateLayerGetter(target_model, return_layers=return_layers)

    def cal_reg_loss(*graph):
        with torch.no_grad():
            layer_outputs_source, _ = source_getter(*graph)
        layer_outputs_target, _ = target_getter(*graph)
        output = 0.0
        count = 0
        for fm_src, fm_tgt in zip(layer_outputs_source.values(), layer_outputs_target.values()):
            delta = fm_tgt - fm_src.detach()
            output = (delta * delta).mean()
            count += 1
        output = output / count
        return output

    return cal_reg_loss


def L2_SP(source_model, target_model):
    initial_named_parameters = source_model.state_dict()

    def cal_reg_loss(*graph):
        loss = 0.0 
        count = 0
        for name, param in target_model.named_parameters():
            initial_param = initial_named_parameters[name]
            delta = param - initial_param
            loss += (delta * delta).mean()
            count += 1
        loss = loss/count
        return loss 
    return cal_reg_loss