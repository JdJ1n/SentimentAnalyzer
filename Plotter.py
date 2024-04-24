import matplotlib.pyplot as plt
import numpy as np


def plot(filepath, tr_data, tst_data, classifier_names, classifier_params, accs, f1_scores):
    plt.ion()
    # Plotting
    x = np.arange(len(classifier_names))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 9))
    rects1 = ax.bar(x - width / 2, accs, width, label='Accuracy')
    rects2 = ax.bar(x + width / 2, f1_scores, width, label='F1 Score')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(f'Scores by classifier {filepath} \n (Train size: {len(tr_data)}, Test size: {len(tst_data)})')
    ax.set_xticks(x)

    # Add classifier parameters to x-axis labels
    ax.set_xticklabels(
        [f'{name}\nParams: {params}' for name, params in zip(classifier_names, classifier_params)],
        wrap=True, fontsize=8)
    plt.xticks(rotation=30)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.5f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=45)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.5f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=45)

    ax.legend()

    fig.tight_layout()

    plt.show(block=True)
