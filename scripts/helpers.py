import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create Logging Directory
if not os.path.exists("/app/logs"):
    os.makedirs("/app/logs")

logger = logging.getLogger('ml_pipeline')  # Set the logger name
handler = logging.FileHandler('/app/logs/helpers.log')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def create_directories(directories):
    for label, path in directories.items():
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created directory for {label}: {path}")
        else:
            logger.info(f"Directory for {label} already exists: {path}")



class PSIModelMonitor:
    def __init__(self, reference_data, model_features, buckets=10):
        self.reference_data = reference_data # Training data
        self.model_features = model_features
        self.buckets = buckets
        self.monitoring_history = []
        
    def calculate_psi(self, expected, actual):
        """Calculate PSI for a single feature"""
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        # Create buckets based on expected data
        breakpoints = np.percentile(expected, np.linspace(0, 100, self.buckets + 1))
        
        # Calculate percentages
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Replace zeros with small value
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi_buckets = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        return np.sum(psi_buckets)
    
    def monitor_drift(self, current_data, monitoring_date=None):
        """Monitor drift for all features"""
        if monitoring_date is None:
            monitoring_date = datetime.now()
            
        feature_psi = {}
        
        for feature in self.model_features:
            if feature in self.reference_data.columns and feature in current_data.columns:
                psi_value = self.calculate_psi(
                    self.reference_data[feature].values,
                    current_data[feature].values
                )
                feature_psi[feature] = psi_value
        
        # Overall dataset PSI (average of features)
        overall_psi = np.mean(list(feature_psi.values()))
        
        # Store monitoring results
        monitoring_result = {
            'date': monitoring_date,
            'overall_psi': overall_psi,
            'feature_psi': feature_psi.copy(),
            'drift_status': self.interpret_psi(overall_psi)
        }
        
        self.monitoring_history.append(monitoring_result)
        return monitoring_result
    
    def interpret_psi(self, psi_value):
        """Interpret PSI value"""
        if psi_value < 0.1:
            return "No significant drift"
        elif psi_value < 0.2:
            return "Moderate drift"
        else:
            return "Significant drift"
    
    def generate_monitoring_report(self, filepath_txt, filepath_fig, last_n_days=30):
        """Generate comprehensive monitoring report"""
        if not self.monitoring_history:
            return "No monitoring data available"
        
        recent_data = self.monitoring_history[-last_n_days:]
        
        with open(filepath_txt, 'w') as f:
            f.write("=== MODEL MONITORING REPORT ===")
            f.write(f"\nMonitoring Period: Last {len(recent_data)} records")
            f.write(f"\nReference Data Size: {len(self.reference_data)}")
        
        # Overall PSI trend
        overall_psi_values = [record['overall_psi'] for record in recent_data]
        dates = [record['date'] for record in recent_data]
        
        plt.figure(figsize=(24, 16))
        
        # Plot 1: Overall PSI trend
        plt.subplot(2, 2, 1)
        plt.plot(dates, overall_psi_values, marker='o', linewidth=2)
        plt.axhline(y=0.1, color='green', linestyle='--', label='Stable threshold')
        plt.axhline(y=0.2, color='red', linestyle='--', label='Drift threshold')
        plt.title('Overall PSI Trend')
        plt.ylabel('PSI Value')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 2: Feature-wise PSI
        plt.subplot(2, 2, 2)
        feature_psi_avg = {}
        for feature in self.model_features:
            feature_values = [record['feature_psi'].get(feature, 0) for record in recent_data]
            feature_psi_avg[feature] = np.mean(feature_values)
        
        features_sorted = sorted(feature_psi_avg.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*features_sorted)
        
        plt.barh(features, values)
        plt.axvline(x=0.1, color='green', linestyle='--')
        plt.axvline(x=0.2, color='red', linestyle='--')
        plt.title('Average Feature PSI (Recent Period)')
        plt.xlabel('Average PSI')
        
        # Plot 3: Drift status distribution
        plt.subplot(2, 2, 3)
        status_counts = {}
        for record in recent_data:
            status = record['drift_status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
        plt.title('Drift Status Distribution')
        
        # Plot 4: Feature correlation with overall drift
        plt.subplot(2, 2, 4)
        correlation_with_overall = {}
        for feature in self.model_features:
            feature_values = [record['feature_psi'].get(feature, 0) for record in recent_data]
            if len(feature_values) >= 1:
                correlation = np.corrcoef(feature_values, overall_psi_values)[0, 1]
                correlation_with_overall[feature] = correlation
        
        correlated_features = sorted(correlation_with_overall.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        features, correlations = zip(*correlated_features)
        
        plt.bar(features, correlations)
        plt.title('Top Features Correlated with Overall Drift')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(filepath_fig)  # Save the image
        plt.close()

        # Print summary statistics
        self._print_summary_statistics(recent_data, filepath_txt)
    
    def _print_summary_statistics(self, recent_data, filepath):
        """Print summary statistics"""
        overall_psi_values = [record['overall_psi'] for record in recent_data]
        
        with open(filepath, 'a') as f:
            f.write(f"\n\nSummary Statistics (Last {len(recent_data)} monitoring periods):")
            f.write(f"\nAverage Overall PSI: {np.mean(overall_psi_values):.4f}")
            f.write(f"\nMax Overall PSI: {np.max(overall_psi_values):.4f}")
            f.write(f"\nMin Overall PSI: {np.min(overall_psi_values):.4f}")
        
            # Alert on high drift features
            f.write("\n\n🚨 High Drift Features (PSI > 0.2):")
            high_drift_features = []
            for feature in self.model_features:
                feature_values = [record['feature_psi'].get(feature, 0) for record in recent_data]
                if np.max(feature_values) > 0.2:
                    high_drift_features.append((feature, np.max(feature_values)))
            
            if high_drift_features:
                for feature, max_psi in sorted(high_drift_features, key=lambda x: x[1], reverse=True):
                    f.write(f"\n - {feature}: Max PSI = {max_psi:.4f}")
            else:
                f.write("\n No high drift features detected")
                
            # Recommendations
            f.write("\n\n Recommendations:")
            max_overall_psi = np.max(overall_psi_values)
            if max_overall_psi > 0.2:
                f.write("\n  Significant drift detected - consider model retraining")
            elif max_overall_psi > 0.1:
                f.write("\n  Moderate drift - monitor closely")
            else:
                f.write("\n  Model is stable - continue monitoring")