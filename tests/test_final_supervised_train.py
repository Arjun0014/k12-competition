import torch

from trace_ace.final_supervised_train import trainable_state_dict


def test_trainable_state_dict_keeps_only_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    state = trainable_state_dict(model)

    assert set(state) == {"1.weight", "1.bias"}
    assert all(tensor.device.type == "cpu" for tensor in state.values())
