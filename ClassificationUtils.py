from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import os.path

from config import GlobalConfig

def describe_test_setup():
    test_str = ""
    if GlobalConfig.get('noise') is not None:
        test_str += ' Noise Invariance {}-{}'.format(GlobalConfig.get('noise'), GlobalConfig.get('noise_val'))
    if GlobalConfig.get('rotate') is not False:
        test_str += ' Rotation Invariance'
    if GlobalConfig.get('test_scale') is not None:
        test_str += ' Scale Invariance Train {} Test {}'.format(int(GlobalConfig.get('scale') * 100), int(GlobalConfig.get('test_scale') * 100))
    return test_str

def pretty_print_conf_matrix(y_true, y_pred,
                             classes,
                             normalize=False,
                             title='{} Confusion matrix'.format(describe_test_setup()),
                             cmap=plt.cm.Blues,
                             out_dir=None):
    """
    Code adapted from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Configure Confusion Matrix Plot Aesthetics (no text yet)
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.colorbar(cax)

    ax.set_ylabel('True label', fontsize=16)
    ax.set_xlabel('Predicted label', fontsize=16, rotation='horizontal')

    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(), 2)  # (axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    # Place Numbers as Text on Confusion Matrix Plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 ha="center",
                 va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    #fig.tight_layout()
    plt.show(block=False)

    if out_dir is not None:
        out_file = os.path.join(out_dir, 'Confusion Matrix{}.png'.format(describe_test_setup()))
        fig.savefig(out_file, dpi=300)


def make_classification_report(y_true, y_pred, classes, out_dir):
    # Get the full report
    rpt = classification_report(y_true, y_pred, labels=classes, output_dict=True)

    # Get the average results from all classes
    rpt_accuracy = rpt['accuracy']
    rpt_macro_avg = rpt['macro avg']
    rpt_weighted_avg = rpt['weighted avg']
    # Get the per-class results only
    del rpt['accuracy']
    del rpt['macro avg']
    del rpt['weighted avg']

    pandas_rpt = pd.DataFrame.from_dict(rpt, orient='index')
    pandas_rpt.rename(columns={'support': 'N Predictions'}, inplace=True)
    pandas_macro_avg = pd.DataFrame.from_dict(rpt_macro_avg, orient='index', columns=['Macro Average'])
    pandas_macro_avg.rename(columns={'support': 'N Predictions'}, inplace=True)
    pandas_weighted_avg = pd.DataFrame.from_dict(rpt_weighted_avg, orient='index', columns=['Weighted Average'])
    pandas_weighted_avg.rename(columns={'support': 'N Predictions'}, inplace=True)
    pandas_accuracy = pd.DataFrame({'': rpt_accuracy}, index=['overall accuracy'])

    out_file = os.path.join(out_dir, 'Classification Report{}.csv'.format(describe_test_setup()))
    pandas_rpt.to_csv(out_file)
    pandas_macro_avg.to_csv(out_file, mode='a', header=True)
    pandas_weighted_avg.to_csv(out_file, mode='a', header=True)
    pandas_accuracy.to_csv(out_file, mode='a', header=True)



