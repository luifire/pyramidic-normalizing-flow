import seaborn as sns
from pyr_flow.utils.evaluation_utils import *

from matplotlib import pyplot as plt


COMPLETE_NLL = 0
TOP_NLL = 1

if __name__ == '__main__':
    results = load_eval_data()

    sns.distplot(results[EvaluatedDatasets.CIFAR_TRAIN][COMPLETE_NLL], label="Cifar Test")
    sns.distplot(results[EvaluatedDatasets.CIFAR_TEST][COMPLETE_NLL], label="Cifar Train")
    sns.distplot(results[EvaluatedDatasets.SVHN][COMPLETE_NLL], label="SVHN")

    # plt.xticks(np.linspace(-11000, 0, 5), fontsize=16)
    plt.yticks(np.linspace(0., 0.0006, 5), fontsize=13)
    plt.legend(fontsize=13)
