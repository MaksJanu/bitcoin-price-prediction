import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def create_data_exploration_plots(df, save_dir='results/plots/lstm'):
    """Tworzy wykresy eksploracji danych - każdy zapisywany osobno"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('default')
    
    price_col = 'Price' if 'Price' in df.columns else 'Close'
    
    if 'Date' in df.columns:
        x_data = df['Date']
        # Convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(x_data):
            x_data = pd.to_datetime(x_data)
    else:
        x_data = df.index
    
    # Wykres 1: Cena bitcoina w czasie
    plt.figure(figsize=(12, 6))
    plt.plot(x_data, df[price_col], linewidth=1, alpha=0.8, color='#1f77b4')
    plt.title('Bitcoin Price Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    price_path = os.path.join(save_dir, 'bitcoin_price_over_time.png')
    plt.savefig(price_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Price chart saved to: {price_path}")
    
    # Wykres 2: Rozkład ceny
    plt.figure(figsize=(10, 6))
    plt.hist(df[price_col], bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Bitcoin Price Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Dodaj statystyki na wykres
    mean_price = df[price_col].mean()
    median_price = df[price_col].median()
    plt.axvline(mean_price, color='red', linestyle='--', label=f'Mean: ${mean_price:.2f}')
    plt.axvline(median_price, color='green', linestyle='--', label=f'Median: ${median_price:.2f}')
    plt.legend()
    plt.tight_layout()
    dist_path = os.path.join(save_dir, 'price_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Price distribution saved to: {dist_path}")

    # Wykres 3: Korelacja między cechami - poprawiona czytelność
    plt.figure(figsize=(14, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Mask dla górnej części macierzy (opcjonalne)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Zwiększ rozmiar i popraw czytelność
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='RdBu_r',  # Lepszy kontrast
                   center=0, 
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={'size': 10},  # Większy tekst
                   linewidths=0.5)  # Linie między komórkami
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        plt.text(0.5, 0.5, 'Not enough numeric columns', 
                ha='center', va='center', fontsize=14)
        plt.title('Correlation Matrix (N/A)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    corr_path = os.path.join(save_dir, 'correlation_matrix.png')
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Correlation matrix saved to: {corr_path}")

    # Wykres 4: Volume w czasie (jeśli dostępne)
    plt.figure(figsize=(12, 6))
    if 'Volume' in df.columns:
        plt.plot(x_data, df['Volume'], label='Volume', alpha=0.7, color='green', linewidth=1)
        plt.title('Bitcoin Trading Volume Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Volume', fontsize=12)
        plt.xticks(rotation=45)
        # Format y-axis dla lepszej czytelności
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plot_name = 'trading_volume.png'
    else:
        # Alternatywnie pokaż High vs Low
        if 'High' in df.columns and 'Low' in df.columns:
            plt.plot(x_data, df['High'], label='High', alpha=0.7, color='red', linewidth=1)
            plt.plot(x_data, df['Low'], label='Low', alpha=0.7, color='blue', linewidth=1)
            plt.fill_between(x_data, df['High'], df['Low'], alpha=0.2, color='gray')
            plt.title('Bitcoin High vs Low Prices', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.legend()
            plt.xticks(rotation=45)
            plot_name = 'high_low_prices.png'
        else:
            plt.text(0.5, 0.5, 'No additional data available', 
                    ha='center', va='center', fontsize=14)
            plt.title('Additional Data (N/A)', fontsize=16, fontweight='bold')
            plot_name = 'no_additional_data.png'
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    volume_path = os.path.join(save_dir, plot_name)
    plt.savefig(volume_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Additional chart saved to: {volume_path}")
    
    # Dodatkowy wykres: Moving Averages (jeśli dostępne)
    if 'MA_7' in df.columns and 'MA_30' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(x_data, df[price_col], label='Price', alpha=0.6, linewidth=1)
        plt.plot(x_data, df['MA_7'], label='MA 7-day', alpha=0.8, linewidth=1.5)
        plt.plot(x_data, df['MA_30'], label='MA 30-day', alpha=0.8, linewidth=1.5)
        plt.title('Bitcoin Price with Moving Averages', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ma_path = os.path.join(save_dir, 'moving_averages.png')
        plt.savefig(ma_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Moving averages chart saved to: {ma_path}")
    
    print(f"\n✅ All data exploration plots saved to: {save_dir}")
    return save_dir

def print_data_statistics(df):
    """Wyświetla podstawowe statystyki danych"""
    price_col = 'Price' if 'Price' in df.columns else 'Close'
    
    print(f"\n📊 Dataset Overview:")
    print(f"Dataset shape: {df.shape}")
    if 'Date' in df.columns:
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Number of features: {len(df.columns)}")
    
    print(f"\n💰 Basic statistics for Bitcoin {price_col}:")
    stats = df[price_col].describe()
    for stat, value in stats.items():
        if stat in ['mean', 'std', 'min', 'max']:
            print(f"  {stat.capitalize()}: ${value:,.2f}")
        else:
            print(f"  {stat}: ${value:,.2f}")
    
    print(f"\n📋 Available columns:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. {col}")
        
    # Sprawdź brakujące wartości
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n⚠️ Missing data found:")
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  {col}: {missing} missing values")
    else:
        print(f"\n✅ No missing data found")