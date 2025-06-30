import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def create_naive_bayes_data_exploration_plots(model, save_dir='results/plots/naive_bayes'):
    """
    Tworzy wykresy eksploracji danych specyficzne dla modelu Naive Bayes
    Pokazuje analizę przetwo rzonych cech statystycznych
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if model.prepared_features_for_correlation is None:
        print("Brak przygotowanych cech. Uruchom najpierw trenowanie modelu.")
        return
    
    # Przygotuj DataFrame z przetworzonymi cechami
    df_features = pd.DataFrame(
        model.prepared_features_for_correlation, 
        columns=model.feature_names
    )
    
    print(f"Analyzing {len(model.feature_names)} processed features for Naive Bayes...")
    
    # 1. Rozkład wartości głównych cech (Price-related)
    plt.figure(figsize=(15, 10))
    price_features = [col for col in df_features.columns if 'Price' in col or 'price' in col]
    
    if price_features:
        n_price_features = len(price_features)
        cols = min(3, n_price_features)
        rows = (n_price_features + cols - 1) // cols
        
        for i, feature in enumerate(price_features):
            plt.subplot(rows, cols, i + 1)
            plt.hist(df_features[feature], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{feature} Distribution', fontsize=12, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No price-related features found', 
                ha='center', va='center', fontsize=14)
        plt.title('Price Features Distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    price_dist_path = os.path.join(save_dir, 'naive_bayes_price_features_distribution.png')
    plt.savefig(price_dist_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Price features distribution saved to: {price_dist_path}")
    
    # 2. Macierz korelacji przetwo rzonych cech
    plt.figure(figsize=(20, 16))
    
    # Oblicz macierz korelacji
    corr_matrix = df_features.corr()
    
    # Wybierz tylko najbardziej istotne cechy (top 20 jeśli jest więcej)
    if len(corr_matrix.columns) > 20:
        # Znajdź cechy z najwyższą średnią korelacją
        mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)
        top_features = mean_corr.head(20).index
        corr_matrix_subset = corr_matrix.loc[top_features, top_features]
        title_suffix = " (Top 20 Features)"
    else:
        corr_matrix_subset = corr_matrix
        title_suffix = ""
    
    # Mask dla górnej części macierzy
    mask = np.triu(np.ones_like(corr_matrix_subset, dtype=bool))
    
    # Heatmap
    sns.heatmap(corr_matrix_subset, 
               mask=mask,
               annot=True, 
               cmap='RdBu_r',
               center=0, 
               square=True, 
               fmt='.2f',
               cbar_kws={"shrink": .8},
               annot_kws={'size': 8})
    
    plt.title(f'Naive Bayes Features Correlation Matrix{title_suffix}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    corr_path = os.path.join(save_dir, 'naive_bayes_features_correlation_matrix.png')
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Features correlation matrix saved to: {corr_path}")
    
    # 3. Analiza rozkładu cech statystycznych
    plt.figure(figsize=(16, 12))
    
    # Grupuj cechy według typu statystyki
    stat_types = ['mean', 'std', 'min', 'max', 'median']
    
    for i, stat_type in enumerate(stat_types):
        plt.subplot(2, 3, i + 1)
        
        stat_features = [col for col in df_features.columns if stat_type in col.lower()]
        if stat_features:
            # Oblicz średnie wartości dla każdej cechy tego typu
            stat_values = []
            stat_names = []
            
            for feature in stat_features:
                stat_values.append(df_features[feature].mean())
                # Wyciągnij nazwę bazowej cechy
                base_name = feature.replace(f'_{stat_type}', '').replace(f'_{stat_type.upper()}', '')
                stat_names.append(base_name)
            
            # Utwórz wykres słupkowy
            bars = plt.bar(range(len(stat_values)), stat_values, alpha=0.7)
            plt.title(f'Average {stat_type.capitalize()} Values', fontsize=12, fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel(f'Average {stat_type}')
            plt.xticks(range(len(stat_names)), [name[:10] + '...' if len(name) > 10 else name for name in stat_names], rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Dodaj wartości na słupkach
            for bar, value in zip(bars, stat_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            plt.text(0.5, 0.5, f'No {stat_type} features', 
                    ha='center', va='center', fontsize=10)
            plt.title(f'{stat_type.capitalize()} Features (N/A)', fontsize=12, fontweight='bold')
    
    # 6th subplot - feature types summary
    plt.subplot(2, 3, 6)
    feature_type_counts = {}
    for stat_type in stat_types:
        count = len([col for col in df_features.columns if stat_type in col.lower()])
        feature_type_counts[stat_type] = count
    
    # Dodaj inne typy cech
    other_features = len([col for col in df_features.columns 
                         if not any(stat in col.lower() for stat in stat_types)])
    if other_features > 0:
        feature_type_counts['other'] = other_features
    
    plt.pie(feature_type_counts.values(), labels=feature_type_counts.keys(), 
            autopct='%1.1f%%', startangle=90)
    plt.title('Feature Types Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    stats_path = os.path.join(save_dir, 'naive_bayes_statistical_features_analysis.png')
    plt.savefig(stats_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Statistical features analysis saved to: {stats_path}")
    
    # 4. Feature importance based on variance
    plt.figure(figsize=(12, 8))
    
    # Oblicz wariancję każdej cechy
    feature_variances = df_features.var().sort_values(ascending=False)
    
    # Pokaż top 15 cech z najwyższą wariancją
    top_var_features = feature_variances.head(15)
    
    bars = plt.barh(range(len(top_var_features)), top_var_features.values)
    plt.yticks(range(len(top_var_features)), 
               [name if len(name) <= 25 else name[:22] + '...' for name in top_var_features.index])
    plt.xlabel('Variance')
    plt.title('Top 15 Features by Variance (Naive Bayes)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Dodaj wartości na słupkach
    for i, (bar, value) in enumerate(zip(bars, top_var_features.values)):
        plt.text(value + max(top_var_features.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    variance_path = os.path.join(save_dir, 'naive_bayes_feature_variance_analysis.png')
    plt.savefig(variance_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Feature variance analysis saved to: {variance_path}")
    
    # 5. Summary statistics table
    plt.figure(figsize=(14, 8))
    plt.axis('off')
    
    # Przygotuj podsumowanie statystyk
    summary_stats = df_features.describe()
    
    # Pokaż tylko wybrane kolumny jeśli jest ich za dużo
    if len(summary_stats.columns) > 10:
        # Wybierz kolumny o najwyższej wariancji
        selected_cols = feature_variances.head(10).index
        summary_stats = summary_stats[selected_cols]
        table_title = "Summary Statistics (Top 10 Features by Variance)"
    else:
        table_title = "Summary Statistics (All Features)"
    
    # Utwórz tabelę
    table_data = []
    for stat in summary_stats.index:
        row = [stat] + [f"{val:.4f}" for val in summary_stats.loc[stat]]
        table_data.append(row)
    
    columns = ['Statistic'] + [col[:15] + '...' if len(col) > 15 else col for col in summary_stats.columns]
    
    table = plt.table(cellText=table_data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Stylizuj tabelę
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title(table_title, fontsize=14, fontweight='bold', pad=20)
    
    summary_path = os.path.join(save_dir, 'naive_bayes_features_summary_statistics.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Features summary statistics saved to: {summary_path}")
    
    print(f"\n✅ Naive Bayes data exploration completed!")
    print(f"📁 All plots saved to: {save_dir}")
    print(f"📊 Total features analyzed: {len(model.feature_names)}")
    print(f"📈 Feature types: {len(stat_types)} statistical types + others")
    
    return {
        'price_distribution_path': price_dist_path,
        'correlation_matrix_path': corr_path,
        'statistical_analysis_path': stats_path,
        'variance_analysis_path': variance_path,
        'summary_statistics_path': summary_path
    }

def print_feature_analysis_summary(model):
    """Wyświetla podsumowanie analizy cech dla modelu Naive Bayes"""
    if model.feature_names is None:
        print("Model nie został jeszcze wytrenowany.")
        return
    
    print("\n" + "="*60)
    print("📊 NAIVE BAYES FEATURES ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total features: {len(model.feature_names)}")
    
    # Analiza typów cech
    feature_types = {}
    stat_types = ['mean', 'std', 'min', 'max', 'median']
    
    for stat_type in stat_types:
        count = len([name for name in model.feature_names if stat_type in name.lower()])
        if count > 0:
            feature_types[stat_type] = count
    
    other_count = len([name for name in model.feature_names 
                      if not any(stat in name.lower() for stat in stat_types)])
    if other_count > 0:
        feature_types['other'] = other_count
    
    print("\nFeature types breakdown:")
    for feat_type, count in feature_types.items():
        print(f"  {feat_type.capitalize()}: {count}")
    
    # Analiza bazowych cech
    base_features = set()
    for name in model.feature_names:
        for stat_type in stat_types:
            if f'_{stat_type}' in name:
                base_features.add(name.replace(f'_{stat_type}', ''))
                break
    
    print(f"\nBase features (original columns): {len(base_features)}")
    if base_features:
        print("  " + ", ".join(sorted(base_features)))
    
    print("="*60)