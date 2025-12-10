
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional


from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
)



FIGURE_SIZE = (8, 6)

def barplot_feature_vs_target(df, feature, target=None, palette=None, top_n=None):
    """
    Draws a bar plot of a feature versus target.
    If target is specified, splits bars by target category.
    Adds annotations for counts and percentages.

    Args:
        df: DataFrame with data.
        feature: Feature column name (categorical).
        target: Target column name (categorical or boolean). Optional.
        palette: Color palette dictionary for target categories.
        top_n: If specified, plot only top N categories of feature.
    """
    data = df.copy()

    # Convert boolean target to string if needed
    if target and data[target].dtype == bool:
        data[target] = data[target].astype(str)

    # Limit to top_n categories if specified
    if top_n:
        top_feats = data[feature].value_counts().nlargest(top_n).index
        data = data[data[feature].isin(top_feats)]

    # Prepare grouped data for plotting
    group_data = data.groupby([feature, target], observed=False).size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    if target:
        ax = sns.barplot(
            x=feature,
            y='count',
            hue=target,
            data=group_data,
            palette=palette,
            errorbar=None
        )

        # Get total counts per feature category for percentages
        total_counts = data.groupby(feature, observed=False).size()

        # Annotate each bar with count and percentage
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                # Get the bar's center x-position
                x = p.get_x() + p.get_width() / 2
                # Determine the feature category label based on the bar position
                # Find the tick label closest to x
                tick_labels = ax.get_xticklabels()
                x_positions = [tick.get_position()[0] for tick in tick_labels]
                # Map x to the closest tick label index
                idx = min(range(len(x_positions)), key=lambda i: abs(x_positions[i] - x))
                feat_label = tick_labels[idx].get_text()

                count = int(height)
                total = total_counts.get(feat_label, 1)  # avoid division by zero
                pct = (count / total) * 100

                # Annotate
                ax.annotate(
                    f"{count}\n({pct:.1f}%)",
                    (x, height),
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='black'
                )
    else:
        # Plot count for feature only
        counts = data[feature].value_counts()
        sns.barplot(x=counts.index, y=counts.values, palette=palette)
        # Add count labels on top
        for idx, count in enumerate(counts.values):
            plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def draw_heatmap(
    data: pd.DataFrame,
    *args,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
 
    fig_size: tuple[int, int] = FIGURE_SIZE,
    **kwargs,
) -> None:
    """
    Create heatmap plot.

    Args:
        data (pd.DataFrame): pandas dataFrame.
        title (Optional[str], optional): Plot title. Defaults to None.
        x_label (Optional[str], optional): x axis label. Defaults to None.
        y_label (Optional[str], optional): Y axis label. Defaults to None.
        fig_size (tuple[int, int], optional): Figure size. Defaults to FIGURE_SIZE.
    """
    plt.figure(figsize=fig_size)
    plt.title(title)
    sns.heatmap(data=data, linewidths=0.5, *args, **kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
   
    plt.show()


def plot_feature_vs_target_and_distribution(
    df, feature, target, hue=None, palette=None, figsize=(12, 4)
    ):
    """
    Creates a figure with:
    - a boxplot of feature vs target (optionally with hue)
    - a distribution plot of the feature
    Plots are side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Boxplot
    sns.boxplot(
        x=target,
        y=feature,
        data=df,
        hue=target,
        palette=palette,
        ax=axes[0],
    )
    title = f"{feature} vs {target}"
    if hue:
        title += f" by {hue}"
    axes[0].set_title(title)
    axes[0].set_xlabel(target)
    axes[0].set_ylabel(feature)

    # Handle legend if hue is used
    if hue and df[hue].nunique() > 1:
        axes[0].legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        if axes[0].get_legend():
            axes[0].get_legend().remove()

    # Distribution of the feature (using histplot for numeric features)
    if pd.api.types.is_numeric_dtype(df[feature]):
        sns.histplot(df[feature], kde=False, ax=axes[1], color='skyblue')
        axes[1].set_title(f"Distribution of {feature}")
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel("Frequency")
    else:
        # If feature is categorical, show barplot of value counts
        counts = df[feature].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=axes[1], color='skyblue')
        axes[1].set_title(f"Distribution of {feature}")
        axes[1].set_xlabel(feature)
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_bar_with_percentage(df, feature_column, palette=None, figsize=(8, 4)):
    """
    Plots a bar chart of the feature with count and percentage annotations on each bar.

    Args:
        df: DataFrame containing the data.
        feature_column: The column name to plot (categorical).
        palette: Optional color palette for the bars.
        figsize: Size of the figure.
    """
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=feature_column, hue=feature_column, data=df, palette=palette)

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            percentage = (height / total) * 100
            ax.annotate(
                f"{height} ({percentage:.1f}%)",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )
    plt.show()




def plot_roc_curve(y_test, y_pred_proba):
    """
    Plots ROC curve for binary classification.

    Args:
        y_test: True binary labels.
        y_pred_proba: Predicted probabilities for the positive class.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix of model {model_name}")
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba):
    """
    Plots Precision-Recall curve for binary classification.

    Args:
        y_test: True binary labels.
        y_pred_proba: Predicted probabilities for the positive class.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"PR curve (area = {avg_precision:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.show()
