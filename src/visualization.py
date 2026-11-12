import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import EDA_PLOTS_DIR, HMM_PLOTS_DIR, PRED_PLOTS_DIR, COMP_PLOTS_DIR, ABLATION_PLOTS_DIR

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 10, 'figure.autolayout': True})

def plot_eda_for_commodity(df: pd.DataFrame, commodity: str):
    """
    Generates 15 required EDA plots for a commodity and saves them under results/plots/eda/.
    """
    comm_dir = EDA_PLOTS_DIR / commodity.lower()
    comm_dir.mkdir(parents=True, exist_ok=True)
    
    dates = pd.to_datetime(df['date'])
    price = df['price']
    temp = df['temperature']
    precip = df['precipitation']
    
    # 1. Price vs Date
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, price, color='#1f77b4', linewidth=1.2, label=f'{commodity} Price (Rs./Qtl)')
    ax.set_title(f'{commodity}: Historical Wholesale Weekly Price (2003–2025)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (Rs. / Quintal)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    fig.savefig(comm_dir / '01_price_vs_date.png', dpi=200)
    plt.close(fig)
    
    # 2. Temperature vs Date
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(dates, temp, color='#ff7f0e', linewidth=1.0, label='Temperature (°C)')
    ax.set_title(f'{commodity} Region: Weekly Mean Temperature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    fig.savefig(comm_dir / '02_temperature_vs_date.png', dpi=200)
    plt.close(fig)
    
    # 3. Precipitation vs Date
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(dates, precip, width=5, color='#2ca02c', alpha=0.7, label='Precipitation (mm)')
    ax.set_title(f'{commodity} Region: Weekly Precipitation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precipitation (mm)')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    fig.savefig(comm_dir / '03_precipitation_vs_date.png', dpi=200)
    plt.close(fig)
    
    # 4. Price Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(price, kde=True, color='#1f77b4', ax=ax, bins=40)
    ax.axvline(price.mean(), color='red', linestyle='--', label=f'Mean: {price.mean():.1f}')
    ax.axvline(price.median(), color='green', linestyle=':', label=f'Median: {price.median():.1f}')
    ax.set_title(f'{commodity}: Price Distribution (Skewness: {price.skew():.2f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Price (Rs./Quintal)')
    ax.legend()
    fig.savefig(comm_dir / '04_price_distribution.png', dpi=200)
    plt.close(fig)
    
    # 5. Temperature Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(temp, kde=True, color='#ff7f0e', ax=ax, bins=30)
    ax.set_title(f'{commodity}: Temperature Distribution (°C)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Temperature (°C)')
    fig.savefig(comm_dir / '05_temperature_distribution.png', dpi=200)
    plt.close(fig)
    
    # 6. Precipitation Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(precip, kde=True, color='#2ca02c', ax=ax, bins=35)
    ax.set_title(f'{commodity}: Precipitation Distribution (mm)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Precipitation (mm)')
    fig.savefig(comm_dir / '06_precipitation_distribution.png', dpi=200)
    plt.close(fig)
    
    # 7. Price vs Temperature
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.regplot(x=temp, y=price, scatter_kws={'alpha':0.4, 's':15}, line_kws={'color':'red'}, ax=ax)
    corr_pt = price.corr(temp)
    ax.set_title(f'{commodity}: Price vs Temperature (Pearson r: {corr_pt:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Price (Rs./Quintal)')
    fig.savefig(comm_dir / '07_price_vs_temperature.png', dpi=200)
    plt.close(fig)
    
    # 8. Price vs Precipitation
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.regplot(x=precip, y=price, scatter_kws={'alpha':0.4, 's':15, 'color':'green'}, line_kws={'color':'blue'}, ax=ax)
    corr_pp = price.corr(precip)
    ax.set_title(f'{commodity}: Price vs Precipitation (Pearson r: {corr_pp:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Price (Rs./Quintal)')
    fig.savefig(comm_dir / '08_price_vs_precipitation.png', dpi=200)
    plt.close(fig)
    
    # 9. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sub_df = df[['price', 'temperature', 'precipitation']].copy()
    sub_df['price_return'] = price.pct_change()
    corr_mat = sub_df.corr()
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f', ax=ax)
    ax.set_title(f'{commodity}: Feature Correlation Heatmap', fontsize=11, fontweight='bold')
    fig.savefig(comm_dir / '09_correlation_heatmap.png', dpi=200)
    plt.close(fig)
    
    # 10. Monthly Price Pattern
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_temp = df.copy()
    df_temp['month'] = dates.dt.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    sns.boxplot(data=df_temp, x='month', y='price', order=month_order, palette='Set3', ax=ax)
    ax.set_title(f'{commodity}: Monthly Wholesale Price Distribution', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price (Rs./Quintal)')
    ax.set_xlabel('Month')
    fig.savefig(comm_dir / '10_monthly_price_pattern.png', dpi=200)
    plt.close(fig)
    
    # 11. Seasonal Price Pattern
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df_temp['quarter'] = 'Q' + dates.dt.quarter.astype(str)
    sns.boxplot(data=df_temp, x='quarter', y='price', palette='Pastel1', ax=ax)
    ax.set_title(f'{commodity}: Quarterly Seasonal Price Pattern', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price (Rs./Quintal)')
    ax.set_xlabel('Quarter')
    fig.savefig(comm_dir / '11_seasonal_price_pattern.png', dpi=200)
    plt.close(fig)
    
    # 12. Yearly Price Pattern
    fig, ax = plt.subplots(figsize=(11, 4.5))
    yearly_avg = df.groupby(dates.dt.year)['price'].agg(['mean', 'median', 'min', 'max'])
    ax.plot(yearly_avg.index, yearly_avg['mean'], marker='o', color='crimson', label='Mean Annual Price')
    ax.plot(yearly_avg.index, yearly_avg['median'], marker='s', color='navy', linestyle='--', label='Median Annual Price')
    ax.fill_between(yearly_avg.index, yearly_avg['min'], yearly_avg['max'], alpha=0.15, color='crimson', label='Annual Min-Max Span')
    ax.set_title(f'{commodity}: Long-Term Annual Price Trend (2003–2025)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price (Rs./Quintal)')
    ax.set_xlabel('Year')
    ax.legend()
    fig.savefig(comm_dir / '12_yearly_price_pattern.png', dpi=200)
    plt.close(fig)
    
    # 13. Rolling Mean (7, 21, 52 weeks)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, price, alpha=0.35, color='gray', label='Raw Weekly Price')
    ax.plot(dates, price.rolling(7).mean(), color='#1f77b4', linewidth=1.2, label='7-week SMA')
    ax.plot(dates, price.rolling(21).mean(), color='#ff7f0e', linewidth=1.4, label='21-week SMA')
    ax.plot(dates, price.rolling(52).mean(), color='#2ca02c', linewidth=1.6, label='52-week SMA (Annual Trend)')
    ax.set_title(f'{commodity}: Price Multi-Scale Rolling Moving Averages', fontsize=11, fontweight='bold')
    ax.set_ylabel('Price (Rs./Quintal)')
    ax.set_xlabel('Date')
    ax.legend()
    fig.savefig(comm_dir / '13_rolling_mean.png', dpi=200)
    plt.close(fig)
    
    # 14. Rolling Volatility (21-week rolling std)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    roll_vol = price.rolling(21).std()
    ax.plot(dates, roll_vol, color='purple', linewidth=1.2, label='21-week Rolling Std Dev')
    ax.axhline(roll_vol.mean(), color='black', linestyle=':', label=f'Mean Volatility: {roll_vol.mean():.1f}')
    ax.set_title(f'{commodity}: 21-Week Rolling Volatility Dynamics', fontsize=11, fontweight='bold')
    ax.set_ylabel('Std Dev (Rs./Quintal)')
    ax.set_xlabel('Date')
    ax.legend()
    fig.savefig(comm_dir / '14_rolling_volatility.png', dpi=200)
    plt.close(fig)
    
    # 15. Price Change / Return Analysis
    fig, ax = plt.subplots(figsize=(12, 3.5))
    weekly_ret = price.pct_change() * 100.0
    ax.plot(dates, weekly_ret, color='teal', linewidth=0.8, alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'{commodity}: Weekly Percentage Price Changes (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Weekly Return (%)')
    ax.set_xlabel('Date')
    fig.savefig(comm_dir / '15_price_change_analysis.png', dpi=200)
    plt.close(fig)
    
    print(f"[OK] Generated all 15 EDA plots for {commodity} in {comm_dir}")
