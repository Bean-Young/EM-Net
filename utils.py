import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, sample_weight=None):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2. * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        if sample_weight is not None:
            loss = loss * sample_weight.unsqueeze(1)
        return loss.mean()

    def forward(self, inputs, target, weight=None, softmax=False, sample_weight=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = torch.ones(self.n_classes, device=inputs.device)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], sample_weight=sample_weight)
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, field, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None,
                       z_spacing=1):
    image, field, label = image.squeeze(0).cpu().detach().numpy(), field.squeeze(
        0).cpu().detach().numpy(), label.squeeze(0).squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        x, y = image.shape[1], image.shape[2]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)
            field = zoom(field, (1, patch_size[0] / x, patch_size[1] / y), order=3)
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        input_f = torch.from_numpy(field).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs, feature  = net(input, input_f)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                prediction = out
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        np.savez(test_save_path + '/' + case + '_pred.npz', array_data=prediction)
    return metric_list
