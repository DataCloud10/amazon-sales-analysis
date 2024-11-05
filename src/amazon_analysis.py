# src/amazon_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AmazonAnalyzer:
    def __init__(self):
        self.data_path = Path('data/amazon_sales.csv')
        self.output_dir = Path('graphs')  # Changed from 'grafici' to 'graphs'
        self.output_dir.mkdir(exist_ok=True)
        
        # Style settings
        plt.style.use('default')
        sns.set_theme()
        
        plt.rcParams.update({
            'figure.figsize': (15, 10),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
        # Color palette
        self.colors = {
            'histogram': '#FF9B9B',
            'bars': sns.color_palette("husl", 15),
            'scatter_price': 'RdYlBu',
            'scatter_rating': 'viridis'
        }
        
        self.INR_TO_EUR = 0.011

    def set_bar_plot_style(self, ax):
        """Sets consistent style for bar plots"""
        ax.grid(True, axis='x', alpha=0.3)
        ax.grid(False, axis='y')
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_axisbelow(True)

    def load_and_clean_data(self):
        """Loads and cleans the dataset"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded. Shape: {df.shape}")
            
            # Price cleaning
            for col in ['actual_price', 'discounted_price']:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('₹', '').str.replace(',', ''), 
                    errors='coerce'
                )
                df[f"{col}_eur"] = df[col] * self.INR_TO_EUR
            
            # Rating and review cleaning
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
            
            # Handle missing values
            df['rating'] = df['rating'].fillna(0)
            df['rating_count'] = df['rating_count'].fillna(0)
            
            # Calculate discount
            df['discount_actual'] = ((df['actual_price'] - df['discounted_price']) / 
                                   df['actual_price'] * 100).round(2)
            
            logger.info("Data cleaning completed")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def save_figure(self, name):
        """Saves figure with error handling"""
        try:
            plt.tight_layout(pad=3.0)
            file_path = self.output_dir / f"{name}.png"
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
            plt.close('all')
            logger.info(f"✓ Saved: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {name}: {e}")
            plt.close('all')
            return False

    def analyze_price_distribution(self, df):
        """Analyzes price distribution"""
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Price distribution
        ax1.hist(df['actual_price_eur'], 
                bins=50, 
                color=self.colors['histogram'],
                edgecolor='black',
                alpha=0.8)
        ax1.set_title('Price Distribution', pad=20)
        ax1.set_xlabel('Price (EUR)')
        ax1.set_ylabel('Number of Products')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Top 15 most expensive categories
        top_prices = df.groupby('category')['actual_price_eur'].mean().nlargest(15)
        top_prices = top_prices.sort_values(ascending=True)
        
        bars = ax2.barh(range(len(top_prices)), 
                       top_prices.values,
                       color=self.colors['bars'],
                       alpha=0.8,
                       height=0.7)
        
        ax2.set_yticks(range(len(top_prices)))
        ax2.set_yticklabels([cat.split('|')[-1] for cat in top_prices.index],
                          fontsize=9)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                    f'€{width:,.2f}',
                    va='center',
                    fontsize=9)
        
        ax2.set_title('Top 15 Most Expensive Categories', pad=20)
        ax2.set_xlabel('Average Price (EUR)')
        
        self.set_bar_plot_style(ax2)
        plt.tight_layout()
        return self.save_figure('price_analysis')

    def analyze_ratings(self, df):
        """Detailed rating analysis"""
        fig = plt.figure(figsize=(15, 14))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # Rating vs Review Count
        scatter = ax1.scatter(df['rating'],
                            df['rating_count'],
                            s=df['rating_count']/df['rating_count'].max()*400,
                            alpha=0.6,
                            c=df['actual_price_eur'],
                            cmap=self.colors['scatter_price'])
        ax1.grid(False)
        plt.colorbar(scatter, ax=ax1, label='Price (EUR)')
        ax1.set_title('Rating vs Review Count', pad=20)
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Review Count')
        ax1.grid(True, alpha=0.3)
        
        # Rating Distribution
        ax2.hist(df[df['rating'] > 0]['rating'],
                bins=np.arange(0, 5.1, 0.1),
                color=self.colors['bars'][2],
                edgecolor='black',
                alpha=0.8)
        ax2.set_title('Detailed Rating Distribution', pad=20)
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Number of Products')
        ax2.grid(True, alpha=0.3)
        
        # Top 15 most reviewed products
        top_reviewed = df.nlargest(15, 'rating_count').iloc[::-1]
        
        bars = ax3.barh(range(len(top_reviewed)),
                       top_reviewed['rating_count'],
                       color=self.colors['bars'],
                       alpha=0.8,
                       height=0.7)
        
        ax3.set_yticks(range(len(top_reviewed)))
        ax3.set_yticklabels([f"{name[:40]}..." for name in top_reviewed['product_name']],
                          fontsize=9)
        
        for i, (bar, rating) in enumerate(zip(bars, top_reviewed['rating'])):
            width = bar.get_width()
            ax3.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                    f'Rating: {rating:.1f}*',
                    va='center',
                    fontsize=9)
            ax3.text(width/2, bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}',
                    va='center',
                    ha='center',
                    fontsize=9)
        
        ax3.set_title('Top 15 Most Reviewed Products', pad=20)
        ax3.set_xlabel('Review Count')
        
        self.set_bar_plot_style(ax3)
        plt.tight_layout()
        return self.save_figure('rating_analysis')

    def analyze_discounts(self, df):
        """Discount analysis"""
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
        # Scatter plot
        scatter = ax1.scatter(df['actual_price_eur'],
                            df['discount_actual'],
                            alpha=0.6,
                            c=df['rating'],
                            cmap=self.colors['scatter_rating'],
                            s=50)
        ax1.grid(False)
        plt.colorbar(scatter, ax=ax1, label='Rating')
        ax1.set_title('Discount vs Price', pad=20)
        ax1.set_xlabel('Price (EUR)')
        ax1.set_ylabel('Discount (%)')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Top 15 categories by discount
        top_discounts = df.groupby('category')['discount_actual'].mean().nlargest(15)
        top_discounts = top_discounts.sort_values(ascending=True)
        
        bars = ax2.barh(range(len(top_discounts)),
                       top_discounts.values,
                       color=self.colors['bars'],
                       alpha=0.8,
                       height=0.7)
        
        ax2.set_yticks(range(len(top_discounts)))
        ax2.set_yticklabels([cat.split('|')[-1] for cat in top_discounts.index],
                          fontsize=9)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width * 1.02, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    va='center',
                    fontsize=9)
        
        ax2.set_title('Top 15 Categories by Average Discount', pad=20)
        ax2.set_xlabel('Average Discount (%)')
        
        self.set_bar_plot_style(ax2)
        plt.tight_layout()
        return self.save_figure('discount_analysis')

def main():
    try:
        # Initialization
        analyzer = AmazonAnalyzer()
        logger.info("Starting Amazon Sales analysis")
        
        # Load data
        df = analyzer.load_and_clean_data()
        
        # Basic statistics
        print("\n=== BASIC STATISTICS ===")
        print(f"Total products: {len(df):,}")
        print(f"Average price: €{df['actual_price_eur'].mean():,.2f}")
        print(f"Median price: €{df['actual_price_eur'].median():,.2f}")
        print(f"Average discount: {df['discount_actual'].mean():.1f}%")
        print(f"Average rating: {df['rating'].mean():.2f}")
        
        # Create visualizations
        logger.info("Generating visualizations...")
        analyzer.analyze_price_distribution(df)
        analyzer.analyze_ratings(df)
        analyzer.analyze_discounts(df)
        
        # Top products
        print("\n=== TOP 5 MOST REVIEWED PRODUCTS ===")
        top_products = df.nlargest(5, 'rating_count')
        for _, row in top_products.iterrows():
            print(f"\nProduct: {row['product_name'][:50]}...")
            print(f"Rating: {row['rating']:.1f}* ({row['rating_count']:,} reviews)")
            print(f"Price: €{row['actual_price_eur']:,.2f}")
            print(f"Discount: {row['discount_actual']:.1f}%")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()