import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class GIDiseaseTrainer:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'Disease_Class'
        
    def load_and_prepare_data(self, file_path):
        """Load Excel data and prepare it for training"""
        print("Loading data...")
        
        try:
            # Load the Excel file
            df = pd.read_excel(file_path)
            print(f"✓ Loaded {len(df)} patient records")
            
            # Display basic info about the dataset
            print(f"✓ Dataset shape: {df.shape}")
            print(f"✓ Features: {len(df.columns)-1}")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\n⚠ Missing values found (will be handled automatically):")
                for col, missing in missing_values[missing_values > 0].items():
                    print(f"  {col}: {missing} missing")
            else:
                print("✓ No missing values found")
                
            return df
            
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found!")
            print("Please make sure your Excel file is in the same directory as this script.")
            return None
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        print("\n" + "="*50)
        print("PREPROCESSING DATA")
        print("="*50)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        
        # Fill missing values
        if len(numeric_columns) > 0:
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        print("✓ Missing values handled")
        
        # Identify categorical columns that need encoding
        categorical_to_encode = ['Gender', 'Obesity_Status', 'Ethnicity', 'Diet_Type', 
                               'Bowel_Habits', 'Disease_Class']
        
        # Encode categorical variables
        df_encoded = df.copy()
        print("\n📊 Encoding categorical variables:")
        
        for col in categorical_to_encode:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                
                # Show encoding mapping
                encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"  {col}: {encoding_map}")
        
        print("✓ Categorical encoding completed")
        return df_encoded
    
    def train_model(self, df):
        """Train the machine learning model"""
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        # Separate features and target
        if self.target_column not in df.columns:
            print(f"❌ Error: Target column '{self.target_column}' not found!")
            print(f"Available columns: {list(df.columns)}")
            return None
            
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        self.feature_columns = X.columns.tolist()
        print(f"✓ Using {len(self.feature_columns)} features for training")
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"\n📈 Disease class distribution:")
        for class_name, count in class_counts.items():
            original_name = self.label_encoders[self.target_column].inverse_transform([class_name])[0]
            print(f"  {original_name}: {count} patients ({count/len(y)*100:.1f}%)")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✓ Data split completed:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        # Train Random Forest model
        print(f"\n🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        self.model.fit(X_train, y_train)
        print("✓ Model training completed!")
        
        # Evaluate the model
        print(f"\n📊 Evaluating model performance...")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"  Training Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
        print(f"  Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        
        if train_accuracy - test_accuracy > 0.1:
            print("  ⚠ Warning: Model might be overfitting (large gap between train/test accuracy)")
        else:
            print("  ✓ Good model performance!")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<25} ({row['importance']:.3f})")
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        target_names = [self.label_encoders[self.target_column].inverse_transform([i])[0] 
                       for i in sorted(y_test.unique())]
        print(classification_report(y_test, test_pred, target_names=target_names))
        
        return True
    
    def save_model(self, filename='gi_disease_model.pkl'):
        """Save the trained model and encoders"""
        print(f"\n💾 Saving model...")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        try:
            joblib.dump(model_data, filename)
            print(f"✓ Model saved successfully as '{filename}'")
            print(f"✓ File size: {round(joblib.load(filename).__sizeof__() / 1024, 2)} KB")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False

def main():
    """Main training function"""
    print("="*60)
    print("    GI DISEASE PREDICTION MODEL TRAINER")
    print("="*60)
    
    # Initialize trainer
    trainer = GIDiseaseTrainer()
    
    # Get dataset file path from user
    print("\n📁 Please provide your dataset file path:")
    print("   (Example: 'gi_disease_dataset.xlsx' or 'data/patients.xlsx')")
    
    while True:
        file_path = input("\nEnter Excel file path: ").strip()
        if file_path:
            break
        print("Please enter a valid file path!")
    
    # Load and prepare data
    df = trainer.load_and_prepare_data(file_path)
    if df is None:
        return
    
    # Preprocess data
    df_processed = trainer.preprocess_data(df)
    
    # Train model
    success = trainer.train_model(df_processed)
    if not success:
        return
    
    # Save model
    print("\n💾 Choose model filename (or press Enter for default):")
    model_filename = input("Model filename [gi_disease_model.pkl]: ").strip()
    if not model_filename:
        model_filename = 'gi_disease_model.pkl'
    
    if trainer.save_model(model_filename):
        print(f"\n🎉 SUCCESS! Model training completed!")
        print(f"✓ Trained model saved as: {model_filename}")
        print(f"✓ You can now use 'predictor.py' to make predictions")
        print(f"\nNext steps:")
        print(f"1. Run: python predictor.py")
        print(f"2. Enter patient data when prompted")
        print(f"3. Get disease predictions!")
    else:
        print(f"\n❌ Failed to save model. Please check file permissions.")

if __name__ == "__main__":
    main()
