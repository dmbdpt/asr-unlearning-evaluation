"""KDE (Kernel Density Estimation) visualization utilities for ML model performance metrics."""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import mlflow

logger = logging.getLogger(__name__)

DATASET_COLORS = {
    'train': '#1f77b4',        # Blue
    'train_no_forget': '#2ca02c',  # Green
    'forget': '#d62728',       # Red
    'validation': '#ff7f0e',  # Orange
    'test': '#9467bd',         # Purple
    'retain': '#8c564b',       # Brown
}

DATASET_DISPLAY_NAMES = {
    'train': 'Train',
    'train_no_forget': 'Train (excl. forget)',
    'forget': 'Forget Set',
    'validation': 'Validation',
    'test': 'Test',
    'retain': 'Retain Set',
}


def organize_metrics_by_category(
    metrics_dict: Dict[str, Any],
    forget_speakers: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """Organize metrics by dataset category from the evaluation results."""
    if forget_speakers is None:
        forget_speakers = []

    forget_set = set(str(s) for s in forget_speakers)

    organized = {
        'train': {},
        'validation': {},
        'test': {},
        'retain': {},
    }

    for set_name, set_data in metrics_dict.items():
        if not isinstance(set_data, dict):
            continue

        for speaker_id, speaker_data in set_data.items():
            if not isinstance(speaker_data, dict):
                continue

            is_forget = str(speaker_id) in forget_set

            if set_name == 'train':
                if is_forget:
                    target_category = 'forget'
                    if 'forget' not in organized:
                        organized['forget'] = {}
                    target_dict = organized['forget']
                else:
                    target_category = 'train_no_forget'
                    if 'train_no_forget' not in organized:
                        organized['train_no_forget'] = {}
                    target_dict = organized['train_no_forget']
            elif set_name == 'validation':
                target_category = 'validation'
                target_dict = organized.get('validation', {})
            elif set_name == 'test':
                target_category = 'test'
                target_dict = organized.get('test', {})
            elif set_name == 'retain':
                target_category = 'retain'
                target_dict = organized.get('retain', {})
            else:
                continue

            if target_category not in organized:
                organized[target_category] = {}

            metrics = speaker_data.get('metrics', {})
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in organized[target_category]:
                            organized[target_category][metric_name] = []
                        organized[target_category][metric_name].append(float(metric_value))

            # 'losses' may be a single scalar or a list of per-utterance values
            losses = speaker_data.get('losses')
            if isinstance(losses, (int, float)):
                metric_name = 'loss'
                if metric_name not in organized[target_category]:
                    organized[target_category][metric_name] = []
                organized[target_category][metric_name].append(float(losses))
            elif isinstance(losses, list):
                metric_name = 'loss'
                if metric_name not in organized[target_category]:
                    organized[target_category][metric_name] = []
                for loss_val in losses:
                    if isinstance(loss_val, (int, float)):
                        organized[target_category][metric_name].append(float(loss_val))

    organized = {k: v for k, v in organized.items() if v}

    return organized


def extract_metric_values(
    metrics_data: Union[Dict, pd.DataFrame],
    category: str,
    metric_name: str
) -> List[float]:
    """Extract metric values for a specific category from metrics data."""
    values = []

    if isinstance(metrics_data, dict):
        category_data = metrics_data.get(category, {})
        if isinstance(category_data, dict):
            for speaker_data in category_data.values():
                if isinstance(speaker_data, dict):
                    if 'metrics' in speaker_data:
                        metrics = speaker_data['metrics']
                        if isinstance(metrics, dict) and metric_name in metrics:
                            val = metrics[metric_name]
                            if isinstance(val, (int, float)):
                                values.append(float(val))
                    elif metric_name in speaker_data:
                        val = speaker_data[metric_name]
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                    elif metric_name == 'loss' and 'losses' in speaker_data:
                        val = speaker_data['losses']
                        if isinstance(val, (int, float)):
                            values.append(float(val))

    elif isinstance(metrics_data, pd.DataFrame):
        # Assume DataFrame has columns: category, metric_name, value
        if 'category' in metrics_data.columns and 'metric_name' in metrics_data.columns:
            subset = metrics_data[
                (metrics_data['category'] == category) &
                (metrics_data['metric_name'] == metric_name)
            ]
            if 'value' in subset.columns:
                values = subset['value'].tolist()
        elif 'set_name' in metrics_data.columns:
            subset = metrics_data[metrics_data['set_name'] == category]
            col_name = f'{metric_name}_pre' if f'{metric_name}_pre' in subset.columns else metric_name
            if col_name in subset.columns:
                values = subset[col_name].dropna().tolist()

    return values


def plot_single_kde(
    data: np.ndarray,
    color: str,
    label: str,
    ax: plt.Axes,
    alpha: float = 0.6
) -> bool:
    """Plot KDE for a single dataset category."""
    if len(data) < 2:
        return False

    try:
        data = data[np.isfinite(data)]

        if len(data) < 2:
            return False

        kde = stats.gaussian_kde(data)

        x_min, x_max = data.min(), data.max()
        x_range = x_max - x_min
        x_padding = x_range * 0.1 if x_range > 0 else 1.0
        x = np.linspace(x_min - x_padding, x_max + x_padding, 500)

        y = kde(x)
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.fill_between(x, y, alpha=alpha, color=color)

        return True

    except Exception as e:
        logger.warning(f"Failed to plot KDE for {label}: {e}")
        return False


def plot_single_vertical_line(
    data: np.ndarray,
    color: str,
    label: str,
    ax: plt.Axes,
    alpha: float = 0.6
) -> bool:
    """Plot a vertical line for single data point categories."""
    if len(data) == 0:
        return False

    value = np.mean(data)

    try:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        ax.axvline(x=value, color=color, linestyle='--', linewidth=2,
                   label=f"{label} (n={len(data)})")

        ax.plot(value, y_max * 0.95, marker='|', markersize=15,
                color=color, markeredgewidth=2)

        return True

    except Exception as e:
        logger.warning(f"Failed to plot vertical line for {label}: {e}")
        return False


def generate_kde_plot(
    category_data: Dict[str, List[float]],
    metric_name: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    show_legend: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.4
) -> plt.Figure:
    """Generate a KDE plot comparing metric values across dataset categories."""
    fig, ax = plt.subplots(figsize=figsize)

    categories_to_plot = []
    for cat in ['train_no_forget', 'forget', 'train', 'validation', 'test', 'retain']:
        if cat in category_data and len(category_data[cat]) > 0:
            categories_to_plot.append(cat)

    if not categories_to_plot:
        ax.text(0.5, 0.5, 'No data available for visualization',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f'{metric_name} Distribution')
        plt.tight_layout()
        return fig

    y_max = 0

    plotted_categories = []
    for category in categories_to_plot:
        data = np.array(category_data[category])

        color = DATASET_COLORS.get(category, '#333333')
        display_name = DATASET_DISPLAY_NAMES.get(category, category)

        if len(data) == 1:
            success = plot_single_vertical_line(data, color, display_name, ax, alpha)
            if success:
                plotted_categories.append(category)
        else:
            success = plot_single_kde(data, color, display_name, ax, alpha)
            if success:
                plotted_categories.append(category)
                try:
                    kde = stats.gaussian_kde(data[np.isfinite(data)])
                    x_min, x_max = data.min(), data.max()
                    x_range = x_max - x_min
                    x_padding = x_range * 0.1 if x_range > 0 else 1.0
                    x = np.linspace(x_min - x_padding, x_max + x_padding, 500)
                    y_vals = kde(x)
                    y_max = max(y_max, y_vals.max())
                except:
                    pass

    if title is None:
        title = f'{metric_name.upper()} Distribution by Dataset Category'
    ax.set_title(title, fontsize=14, fontweight='bold')

    if xlabel is None:
        xlabel = metric_name
    ax.set_xlabel(xlabel, fontsize=12)

    ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    if show_legend and plotted_categories:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved KDE plot to {output_path}")

    return fig


def generate_all_kde_plots(
    metrics_by_category: Dict[str, Dict[str, List[float]]],
    output_dir: Optional[str] = None,
    log_to_mlflow: bool = True,
    artifact_path: str = "kde_plots",
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.4
) -> Dict[str, plt.Figure]:
    """Generate KDE plots for all metrics across all dataset categories."""
    figures = {}

    all_metrics = set()
    for category_data in metrics_by_category.values():
        if isinstance(category_data, dict):
            all_metrics.update(category_data.keys())

    if not all_metrics:
        logger.warning("No metrics found to visualize")
        return figures

    for metric_name in sorted(all_metrics):
        category_data = {}
        for category, metrics_dict in metrics_by_category.items():
            if isinstance(metrics_dict, dict) and metric_name in metrics_dict:
                values = metrics_dict[metric_name]
                if values:
                    category_data[category] = values

        if not category_data:
            logger.warning(f"No data found for metric: {metric_name}")
            continue

        output_path = None
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'kde_{metric_name}.png')

        fig = generate_kde_plot(
            category_data=category_data,
            metric_name=metric_name,
            output_path=output_path,
            figsize=figsize,
            alpha=alpha
        )

        figures[metric_name] = fig

        if log_to_mlflow:
            try:
                mlflow.log_figure(fig, f"{artifact_path}/kde_{metric_name}.png")
                logger.info(f"Logged KDE plot for {metric_name} to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log {metric_name} plot to MLflow: {e}")

    return figures


def plot_metrics_kde(
    metrics_dict: Dict[str, Any],
    forget_speakers: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    log_to_mlflow: bool = True,
    artifact_path: str = "kde_plots",
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.4
) -> Dict[str, plt.Figure]:
    """KDE plots per metric; forget_speakers splits 'train' into 'train_no_forget' + 'forget'."""
    if not metrics_dict:
        logger.warning("Empty metrics dictionary provided")
        return {}

    metrics_by_category = organize_metrics_by_category(metrics_dict, forget_speakers)

    if not metrics_by_category:
        logger.warning("No valid metrics found in the input data")
        return {}

    for category, metrics in metrics_by_category.items():
        for metric_name, values in metrics.items():
            logger.info(f"Category '{category}', Metric '{metric_name}': {len(values)} samples")

    figures = generate_all_kde_plots(
        metrics_by_category=metrics_by_category,
        output_dir=output_dir,
        log_to_mlflow=log_to_mlflow,
        artifact_path=artifact_path,
        figsize=figsize,
        alpha=alpha
    )

    return figures


def plot_loss_distribution(
    loss_results: Dict[str, Any],
    forget_speakers: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    log_to_mlflow: bool = True,
    artifact_path: str = "kde_plots"
) -> Dict[str, plt.Figure]:
    """Generate KDE plots specifically for loss values."""
    return plot_metrics_kde(
        metrics_dict=loss_results,
        forget_speakers=forget_speakers,
        output_dir=output_dir,
        log_to_mlflow=log_to_mlflow,
        artifact_path=artifact_path
    )


def create_comparison_summary(
    metrics_by_category: Dict[str, Dict[str, List[float]]]
) -> pd.DataFrame:
    """Create a summary DataFrame comparing metric statistics across categories."""
    rows = []

    for category, metrics_dict in metrics_by_category.items():
        if not isinstance(metrics_dict, dict):
            continue

        for metric_name, values in metrics_dict.items():
            if not values:
                continue

            values_array = np.array(values)
            values_array = values_array[np.isfinite(values_array)]

            if len(values_array) == 0:
                continue

            rows.append({
                'category': category,
                'metric': metric_name,
                'count': len(values_array),
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'median': np.median(values_array)
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def log_summary_to_mlflow(
    metrics_by_category: Dict[str, Dict[str, List[float]]],
    artifact_path: str = "kde_plots"
) -> None:
    """Log summary statistics to MLflow as a table."""
    summary_df = create_comparison_summary(metrics_by_category)

    if summary_df.empty:
        logger.warning("No summary data to log to MLflow")
        return

    try:
        mlflow.log_table(
            data=summary_df,
            artifact_file=f"{artifact_path}/kde_summary.json"
        )
        logger.info("Logged KDE summary statistics to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log summary to MLflow: {e}")
