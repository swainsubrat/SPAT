"""
file containing all the necessary function
to plot bar and line charts
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_lips(attacks, lips, semantic_lips, save_path=None):
    """
    Plots a bar graph comparing the LPIPS score for standard and semantic adversarial attacks.

    Parameters:
    -----------
    attacks: list
        A list of strings representing the names of the attack methods to be plotted.
    lips: list
        A list of floats representing the LPIPS score for the standard adversarial attacks.
    semantic_lips: list
        A list of floats representing the LPIPS score for the semantic adversarial attacks.
    save_path: str, optional
        A string representing the file name to save the plot. If not given, the plot will not be saved.

    Returns:
    --------
    None
    """
    ind = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, lips, width, color='red', label='X')
    rects2 = ax.bar(ind + width/2, semantic_lips, width, color='green', label='Semantic-X')

    ax.set_ylabel('LPIPS Distance')
    ax.set_xlabel('Attack Methods')
    ax.set_title('Comparision of LPIPS Score for X and Semantic-X Attacks on CIFAR-10')
    ax.set_xticks(ind)
    ax.set_xticklabels(attacks)
    ax.legend()

    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.show()

def plot_robust_accuracy(attacks, robust_acc, semantic_robust_acc, save_path=None):
    """
    Plot the comparison of robust accuracy for X and Semantic-X attacks on CIFAR-10.

    Parameters:
    -----------
    attacks : list
        A list of attack names to be used as xticks.
    robust_acc : list
        A list of robust accuracy scores for the X attacks.
    semantic_robust_acc : list
        A list of robust accuracy scores for the Semantic-X attacks.
    save_path : str or None, optional
        If provided, the plot will be saved to this path instead of displaying it.

    Returns:
    --------
    None
    """
    ind = np.arange(len(attacks))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, robust_acc, width, color='red', label='X')
    rects2 = ax.bar(ind + width/2, semantic_robust_acc, width, color='green', label='Semantic-X')

    ax.set_ylabel('Robust Accuracy')
    ax.set_xlabel('Attack Methods')
    ax.set_title('Comparison of Robust Accuracy for X and Semantic-X Attacks on CIFAR-10')
    ax.set_xticks(ind)
    ax.set_xticklabels(attacks)
    ax.legend()

    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # attacks = ['FGSM', 'PGD', 'DeepFool', 'C&W', 'EAD']
    # lips = [0.0411, 0.0207, 0.1531, 0.1492, 0.2334]
    # semantic_lips = [0.0183, 0.0176, 0.0092, 0.0086, 0.0837]

    # plot_lips(attacks, lips, semantic_lips, save_path='cifar10_lpips.png')

    attacks = ['FGSM', 'PGD', 'DeepFool', 'C&W', 'EAD']
    accuracies = [0.23, 0.01, 0.04, 0.01, 0.09]
    semantic_accuracies = [0.22, 0.03, 0.03, 0.04, 0.01]
    plot_robust_accuracy(attacks, accuracies, semantic_accuracies, save_path='cifar10_ra.png')
