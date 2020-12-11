from pyr_flow.constants import *
from pyr_flow.constants import *
from pyr_flow.model.pyramid_flow_model import PyramidFlowModel
from pyr_flow.utils.data_utils import load_dataset
from pyr_flow.model.computation_layers.pyramid_loss import PyramidLoss

def _print_state(batch_idx, dataset_loader):
    percentage = 100. * batch_idx / len(dataset_loader)
    print(f'{percentage:.2f}%')


def eval_all(model, loss, dataset_loader):
    nll_list = []
    top_nll_list = []

    for batch_idx, (data, target) in enumerate(dataset_loader):
        if batch_idx % LOG_INTERVAL:
            _print_state(batch_idx, dataset_loader)

            data = data.to(DEVICE)

            pyramid_steps, pyramid_steps_lnorm = model(data)
            _, nll, _, top_nll = loss(pyramid_steps, pyramid_steps_lnorm)
            nll_list.append(nll.detach().cpu())
            top_nll_list.append(top_nll.detach().cpu())

    np_nll_list = torch.cat(nll_list).numpy()
    np_top_list = torch.cat(top_nll_list).numpy()

    _print_state(len(dataset_loader), dataset_loader)

    return np_nll_list, np_top_list


class EvaluatedDatasets(Enum):
    CIFAR_TEST = 1
    CIFAR_TRAIN = 2
    SVHN = 3


def load_eval_data(model_path):

    model = PyramidFlowModel()
    model.eval()
    model.load_state_dict(torch.load(model_path))

    loss = PyramidLoss()
    loss.eval()

    # Load Data
    cifar_loader_train, cifar_loader_test = load_dataset(DataSet.CIFAR, BATCH_SIZE_TEST, BATCH_SIZE_TEST)

    print('eval cifar test')
    result_cifar_test = eval_all(model, loss, cifar_loader_test)
    print('eval cifar train')
    result_cifar_train = eval_all(model, loss, cifar_loader_train)

    svhn_loader, _ = load_dataset(DataSet.SVHN, BATCH_SIZE_TEST, BATCH_SIZE_TEST)
    print('eval svhn train')
    result_svhn = eval_all(model, loss, svhn_loader)

    return {EvaluatedDatasets.CIFAR_TEST: result_cifar_test,
            EvaluatedDatasets.CIFAR_TRAIN: result_cifar_train,
            EvaluatedDatasets.SVHN: result_svhn}