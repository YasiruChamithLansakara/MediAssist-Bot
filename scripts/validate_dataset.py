import pandas as pd
import numpy as np
from collections import Counter
import json

class DatasetValidator:
    def __init__(self, csv_path):
        self.df = pd.read_csv("data/processed/drug_knowledge_bot_ready_clean.csv")
        self.validation_report = {}
    
    def check_missing_values(self):
        """Check for missing values in each column"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        
        report = {
            "column": missing.index.tolist(),
            "count": missing.values.tolist(),
            "percentage": missing_pct.values.tolist()
        }
        
        self.validation_report['missing_values'] = report
        print("❌ Missing Values Report:")
        print(pd.DataFrame(report))
        return report
    
    def check_duplicates(self):
        """Check for duplicate drug entries"""
        duplicates = self.df.duplicated(subset=['drug_id', 'generic_name']).sum()
        
        self.validation_report['duplicates'] = {
            "total_duplicates": duplicates,
            "percentage": (duplicates / len(self.df)) * 100
        }
        
        print(f"🔄 Duplicate Records: {duplicates}")
        return duplicates
    
    def check_column_types(self):
        """Verify column data types"""
        print("\n📋 Column Data Types:")
        print(self.df.dtypes)
        self.validation_report['column_types'] = self.df.dtypes.astype(str).to_dict()
    
    def check_data_completeness(self):
        """Check completeness by row"""
        # Calculate percentage of non-null values per row
        completeness = (self.df.notna().sum(axis=1) / len(self.df.columns)) * 100
        
        report = {
            "mean_completeness": completeness.mean(),
            "min_completeness": completeness.min(),
            "max_completeness": completeness.max(),
            "rows_with_<80%_data": (completeness < 80).sum()
        }
        
        self.validation_report['completeness'] = report
        print(f"\n📊 Data Completeness:")
        print(f"   Mean: {report['mean_completeness']:.2f}%")
        print(f"   Min: {report['min_completeness']:.2f}%")
        print(f"   Rows with <80% data: {report['rows_with_<80%_data']}")
        
        return report
    
    def check_critical_fields(self):
        """Ensure critical fields are populated"""
        critical_fields = ['drug_id', 'generic_name', 'indications', 'warnings']
        
        report = {}
        for field in critical_fields:
            populated = self.df[field].notna().sum()
            report[field] = {
                "populated": populated,
                "empty": len(self.df) - populated,
                "percentage": (populated / len(self.df)) * 100
            }
        
        self.validation_report['critical_fields'] = report
        print(f"\n🔑 Critical Fields Status:")
        for field, stats in report.items():
            print(f"   {field}: {stats['percentage']:.2f}% populated")
        
        return report
    
    def generate_report(self, output_path='validation_report.json'):
        """Generate comprehensive validation report"""
        self.check_missing_values()
        self.check_duplicates()
        self.check_column_types()
        self.check_data_completeness()
        self.check_critical_fields()
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2, default=str)
        
        print(f"\n✅ Validation report saved to {output_path}")
        return self.validation_report

# Usage
if __name__ == "__main__":
    validator = DatasetValidator('data/processed/merged_medicine_dataset.csv')
    validator.generate_report()