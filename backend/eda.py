"""
eda.py — Exploratory Data Analysis Module
Generates comprehensive visualizations for the IFF Scent Duration dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import parse_ingredients, ALL_INGREDIENTS, INGREDIENT_NOTE_MAP

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')


def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_target_distribution(df):
    """Plot ScentDuration_Hours distribution (raw + log-transformed)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw distribution
    axes[0].hist(df['ScentDuration_Hours'], bins=30, color='#2196F3', edgecolor='white', alpha=0.85)
    axes[0].axvline(df['ScentDuration_Hours'].mean(), color='red', linestyle='--', label=f"Mean: {df['ScentDuration_Hours'].mean():.1f}h")
    axes[0].axvline(df['ScentDuration_Hours'].median(), color='orange', linestyle='--', label=f"Median: {df['ScentDuration_Hours'].median():.1f}h")
    axes[0].set_xlabel('Scent Duration (Hours)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Scent Duration Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()

    # Log-transformed
    log_duration = np.log1p(df['ScentDuration_Hours'])
    axes[1].hist(log_duration, bins=30, color='#4CAF50', edgecolor='white', alpha=0.85)
    axes[1].axvline(log_duration.mean(), color='red', linestyle='--', label=f"Mean: {log_duration.mean():.2f}")
    axes[1].set_xlabel('log(1 + Duration)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Log-Transformed Duration', fontsize=14, fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'target_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved target_distribution.png")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap of all numerical features."""
    numeric_cols = ['Concentration_Pct', 'TopNotes_Pct', 'MiddleNotes_Pct', 'BaseNotes_Pct',
                    'NumIngredients', 'AvgMolecularWeight', 'VolatilityScore', 'LogP',
                    'VaporPressure_Pa', 'BoilingPoint_C', 'ScentDuration_Hours']

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True, fmt='.2f',
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved correlation_heatmap.png")


def plot_duration_by_category(df):
    """Box plots of duration by FragranceFamily and ApplicationType."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # By Fragrance Family
    family_order = df.groupby('FragranceFamily')['ScentDuration_Hours'].median().sort_values().index
    sns.boxplot(data=df, x='FragranceFamily', y='ScentDuration_Hours', order=family_order,
                ax=axes[0], palette='viridis')
    axes[0].set_title('Duration by Fragrance Family', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fragrance Family', fontsize=12)
    axes[0].set_ylabel('Duration (Hours)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=30)

    # By Application Type
    app_order = df.groupby('ApplicationType')['ScentDuration_Hours'].median().sort_values().index
    sns.boxplot(data=df, x='ApplicationType', y='ScentDuration_Hours', order=app_order,
                ax=axes[1], palette='magma')
    axes[1].set_title('Duration by Application Type', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Application Type', fontsize=12)
    axes[1].set_ylabel('Duration (Hours)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'duration_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved duration_by_category.png")


def plot_concentration_vs_duration(df):
    """Scatter plot of the dominant feature vs target."""
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(df['Concentration_Pct'], df['ScentDuration_Hours'],
                         c=df['ScentDuration_Hours'], cmap='RdYlGn', alpha=0.6,
                         edgecolors='white', linewidth=0.5, s=50)

    # Trend line
    z = np.polyfit(df['Concentration_Pct'], df['ScentDuration_Hours'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Concentration_Pct'].min(), df['Concentration_Pct'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8,
            label=f'Linear fit (r = {df["Concentration_Pct"].corr(df["ScentDuration_Hours"]):.3f})')

    plt.colorbar(scatter, label='Duration (h)', ax=ax)
    ax.set_xlabel('Concentration (%)', fontsize=13)
    ax.set_ylabel('Scent Duration (Hours)', fontsize=13)
    ax.set_title('Concentration vs. Scent Duration', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'concentration_vs_duration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved concentration_vs_duration.png")


def plot_longevity_distribution(df):
    """Pie + bar chart of LongevityCategory distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cat_order = ['Short', 'Medium', 'Long', 'Very Long']
    cat_counts = df['LongevityCategory'].value_counts().reindex(cat_order)
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']

    # Pie chart
    axes[0].pie(cat_counts, labels=cat_order, autopct='%1.1f%%', colors=colors,
                startangle=90, textprops={'fontsize': 11})
    axes[0].set_title('Longevity Category Distribution', fontsize=14, fontweight='bold')

    # Bar chart with counts
    bars = axes[1].bar(cat_order, cat_counts, color=colors, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, cat_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     str(count), ha='center', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longevity Category', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Sample Count per Category', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'longevity_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved longevity_distribution.png")


def plot_ingredient_frequency(df):
    """Bar chart of ingredient frequencies colored by note type."""
    all_parsed = df['Ingredients'].apply(parse_ingredients)
    from collections import Counter
    counter = Counter()
    for ing_list in all_parsed:
        counter.update(ing_list)

    # Sort by frequency
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_items]
    counts = [x[1] for x in sorted_items]
    colors = []
    for name in names:
        note_type = INGREDIENT_NOTE_MAP.get(name, 'Unknown')
        if note_type == 'Top':
            colors.append('#e74c3c')
        elif note_type == 'Middle':
            colors.append('#f39c12')
        else:
            colors.append('#27ae60')

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Ingredient Frequency (Red=Top, Orange=Middle, Green=Base)', fontsize=14, fontweight='bold')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ingredient_frequency.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved ingredient_frequency.png")


def plot_note_violins(df):
    """Violin plots for note percentages across longevity categories."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cat_order = ['Short', 'Medium', 'Long', 'Very Long']

    for i, (col, title) in enumerate([
        ('TopNotes_Pct', 'Top Notes %'),
        ('MiddleNotes_Pct', 'Middle Notes %'),
        ('BaseNotes_Pct', 'Base Notes %')
    ]):
        sns.violinplot(data=df, x='LongevityCategory', y=col, order=cat_order,
                       ax=axes[i], palette='coolwarm', inner='box')
        axes[i].set_title(title, fontsize=13, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Percentage', fontsize=11)

    plt.suptitle('Note Composition by Longevity Category', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'note_violins.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved note_violins.png")


def plot_chemistry_features(df):
    """Scatter matrix of key chemistry features vs duration."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    features = [
        ('VolatilityScore', 'Volatility Score'),
        ('LogP', 'LogP (Lipophilicity)'),
        ('VaporPressure_Pa', 'Vapor Pressure (Pa)'),
        ('BoilingPoint_C', 'Boiling Point (°C)'),
        ('AvgMolecularWeight', 'Avg Molecular Weight'),
        ('NumIngredients', 'Num Ingredients'),
    ]

    for idx, (col, label) in enumerate(features):
        ax = axes[idx // 3][idx % 3]
        ax.scatter(df[col], df['ScentDuration_Hours'], alpha=0.4, s=30,
                   c=df['Concentration_Pct'], cmap='plasma')
        r = df[col].corr(df['ScentDuration_Hours'])
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Duration (h)', fontsize=11)
        ax.set_title(f'{label} (r={r:.3f})', fontsize=12, fontweight='bold')

    plt.suptitle('Chemistry Features vs. Scent Duration', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'chemistry_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[EDA] ✓ Saved chemistry_features.png")


def run_full_eda(df):
    """Generate all EDA plots."""
    ensure_plot_dir()
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_duration_by_category(df)
    plot_concentration_vs_duration(df)
    plot_longevity_distribution(df)
    plot_ingredient_frequency(df)
    plot_note_violins(df)
    plot_chemistry_features(df)

    print("=" * 60)
    print(f"[EDA] ✅ All plots saved to {PLOT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    from data_loader import load_dataset
    df = load_dataset()
    run_full_eda(df)
