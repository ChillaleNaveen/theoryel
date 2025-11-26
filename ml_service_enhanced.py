"""
Enhanced ML + Explainable AI Service for LiveInsight+
FastAPI service with Random Forest predictions
Implements BOTH SHAP (global) and LIME (local agent decisions) for comprehensive XAI
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ML and XAI libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import shap
import lime
import lime.lime_tabular
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LiveInsight+ Enhanced ML & XAI Service",
    description="Inventory prediction with SHAP (global) and LIME (agent decisions)",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directories
os.makedirs('output', exist_ok=True)
os.makedirs('output/xai', exist_ok=True)
os.makedirs('output/xai/shap', exist_ok=True)
os.makedirs('output/xai/lime', exist_ok=True)
os.makedirs('models', exist_ok=True)


# Request/Response models
class PredictionRequest(BaseModel):
    product: str
    current_stock: int
    avg_daily_sales: float
    
class PredictionResponse(BaseModel):
    product: str
    predicted_days_to_depletion: float
    confidence: float
    recommendation: str
    urgency_level: str
    
class SHAPExplanationRequest(BaseModel):
    product: str
    
class LIMEExplanationRequest(BaseModel):
    product: str
    current_stock: float
    avg_daily_sales: float
    stock_percentage: float
    decision_type: str = "reorder"  # reorder, urgency, quantity
    
class ExplanationResponse(BaseModel):
    product: str
    explanation_type: str  # "SHAP" or "LIME"
    feature_contributions: Dict[str, float]
    base_value: float
    prediction_value: float
    explanation_text: str
    decision_rationale: Optional[str] = None


class EnhancedInventoryMLService:
    """
    Enhanced ML service with Random Forest and dual XAI
    """
    
    def __init__(self):
        self.model = None
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = ['CurrentStock', 'AvgDailySales', 'StockPercentage']
        self.training_data = None  # Keep for LIME
        
        # Paths
        self.model_path = 'models/rf_inventory_model.pkl'
        self.shap_explainer_path = 'models/shap_explainer.pkl'
        self.lime_explainer_path = 'models/lime_explainer.pkl'
        
        # Training metadata
        self.metadata = {
            'trained_at': None,
            'n_samples': 0,
            'n_features': len(self.feature_names),
            'mae': 0.0,
            'rmse': 0.0,
            'r2': 0.0,
            'feature_importance': {},
            'model_type': 'RandomForestRegressor'
        }
        
    def prepare_training_data(self) -> Optional[pd.DataFrame]:
        """
        Load and prepare training data from processor output
        Uses REAL sales and inventory data from processor_consumer.py
        """
        try:
            df = pd.read_csv('output/product_inventory_usage.csv')
            
            if len(df) < 1:
                logger.warning(f"Insufficient data: {len(df)} products")
                return None
            
            # Clean data
            df['TotalUnitsSold'] = pd.to_numeric(df['TotalUnitsSold'], errors='coerce').fillna(0)
            
            # Use REAL CurrentStock from the CSV if available, otherwise fallback (but it should be there now)
            if 'CurrentStock' in df.columns:
                 df['CurrentStock'] = pd.to_numeric(df['CurrentStock'], errors='coerce').fillna(100)
            else:
                 # Fallback only if CSV structure is old
                 df['CurrentStock'] = 1000 - df['TotalUnitsSold']
            
            # Assume Average Daily Sales based on Total Sold (assuming approx 5 days of data for rate calculation)
            # In production, you'd divide by actual days elapsed
            df['AvgDailySales'] = df['TotalUnitsSold'] / 5.0 
            df['AvgDailySales'] = df['AvgDailySales'].replace(0, 0.1) # Avoid divide by zero
            
            # Calculate Starting Stock (Current + Sold) to get Percentage
            df['StartingStock'] = df['CurrentStock'] + df['TotalUnitsSold']
            df['StockPercentage'] = (df['CurrentStock'] / df['StartingStock']) * 100
            df['StockPercentage'] = df['StockPercentage'].fillna(100) # Handle new items
            
            # Target: Days to depletion
            df['DaysToDepletion'] = df.apply(
                lambda row: row['CurrentStock'] / row['AvgDailySales'] 
                if row['AvgDailySales'] > 0 else 999,
                axis=1
            )
            
            # Add slight noise for training robustness, but keep logic grounded in reality
            noise = np.random.normal(0, 0.5, len(df))
            df['DaysToDepletion'] = (df['DaysToDepletion'] + noise).clip(lower=0)
            
            # Clean outliers
            df = df[df['DaysToDepletion'] < 1000]
            
            logger.info(f"✅ Prepared {len(df)} training samples with {len(self.feature_names)} features")
            return df
            
        except FileNotFoundError:
            logger.warning("product_inventory_usage.csv not found - waiting for streaming data")
            return None
        except Exception as e:
            logger.error(f"Error preparing data: {e}", exc_info=True)
            return None
    
    def train_model(self) -> bool:
        """
        Train Random Forest model with SHAP and LIME explainers
        """
        try:
            df = self.prepare_training_data()
            if df is None:
                return False
            
            # Features and target
            X = df[self.feature_names]
            y = df['DaysToDepletion']
            
            # If not enough samples for split, just train on all
            if len(df) < 5:
                X_train, X_test, y_train, y_test = X, X, y, y
            else:
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Train Random Forest
            logger.info("🧠 Training Random Forest Regressor...")
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=2, # Adjusted for small data
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred_test = self.model.predict(X_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2_test = r2_score(y_test, y_pred_test)
            
            logger.info("=" * 70)
            logger.info("📊 MODEL PERFORMANCE METRICS")
            logger.info(f"   Test MAE:  {mae_test:.2f} days")
            logger.info(f"   Test RMSE: {rmse_test:.2f} days")
            logger.info("=" * 70)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Create Explainers
            self.shap_explainer = shap.TreeExplainer(self.model)
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=self.feature_names,
                mode='regression',
                verbose=False
            )
            
            # Store training data
            self.training_data = X_train
            
            # Update metadata
            self.metadata = {
                'trained_at': datetime.now().isoformat(),
                'n_samples': len(df),
                'n_features': len(self.feature_names),
                'mae': float(mae_test),
                'rmse': float(rmse_test),
                'r2': float(r2_test),
                'feature_importance': {k: float(v) for k, v in feature_importance.items()}
            }
            
            # Save artifacts
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.shap_explainer, self.shap_explainer_path)
            joblib.dump(self.lime_explainer, self.lime_explainer_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return False
    
    def load_model(self) -> bool:
        """Load pre-trained model and explainers"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                # Try loading explainers if they exist
                if os.path.exists(self.shap_explainer_path):
                    self.shap_explainer = joblib.load(self.shap_explainer_path)
                if os.path.exists(self.lime_explainer_path):
                    self.lime_explainer = joblib.load(self.lime_explainer_path)
                
                logger.info("✅ Loaded pre-trained Random Forest model")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, current_stock: int, avg_daily_sales: float) -> Dict:
        """Predict days to depletion"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Calculate stock percentage estimates
        # (Approximate starting stock as Current + (Sales * 5 days))
        estimated_starting = current_stock + (avg_daily_sales * 5)
        stock_percentage = (current_stock / estimated_starting * 100) if estimated_starting > 0 else 0
        
        X = pd.DataFrame([[current_stock, avg_daily_sales, stock_percentage]], 
                        columns=self.feature_names)
        
        # Predict
        days_pred = self.model.predict(X)[0]
        
        # Confidence
        predictions_per_tree = [tree.predict(X)[0] for tree in self.model.estimators_]
        std_dev = np.std(predictions_per_tree)
        confidence = 1.0 / (1.0 + std_dev)
        
        # Urgency
        if days_pred < 3:
            urgency = "CRITICAL"
            recommendation = "🚨 CRITICAL: Immediate reorder required!"
        elif days_pred < 7:
            urgency = "HIGH"
            recommendation = "⚠️  HIGH: Reorder within 24 hours"
        elif days_pred < 14:
            urgency = "MEDIUM"
            recommendation = "📊 MEDIUM: Plan reorder this week"
        else:
            urgency = "LOW"
            recommendation = "✅ LOW: Monitor and plan ahead"
        
        return {
            'predicted_days': float(days_pred),
            'confidence': float(confidence),
            'recommendation': recommendation,
            'urgency_level': urgency,
            'features_used': {
                'current_stock': current_stock,
                'avg_daily_sales': avg_daily_sales,
                'stock_percentage': stock_percentage
            }
        }
    
    def explain_with_shap(self, current_stock: int, avg_daily_sales: float, 
                          stock_percentage: float) -> Dict:
        """Generate SHAP explanation"""
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not available")
        
        X = pd.DataFrame([[current_stock, avg_daily_sales, stock_percentage]], 
                        columns=self.feature_names)
        
        shap_values = self.shap_explainer.shap_values(X)[0]
        base_value = self.shap_explainer.expected_value
        prediction = self.model.predict(X)[0]
        
        contributions = dict(zip(self.feature_names, shap_values))
        
        explanations = []
        for feature, value in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(value) > 0.1:
                direction = "extends" if value > 0 else "reduces"
                explanations.append(
                    f"{feature} {direction} timeline by {abs(value):.1f} days"
                )
        
        explanation_text = "; ".join(explanations) if explanations else "Minor impact from all features"
        
        return {
            'explanation_type': 'SHAP',
            'feature_contributions': {k: float(v) for k, v in contributions.items()},
            'base_value': float(base_value),
            'prediction_value': float(prediction),
            'explanation_text': explanation_text,
            'feature_importance': self.metadata.get('feature_importance', {})
        }
    
    def explain_with_lime(self, current_stock: float, avg_daily_sales: float, 
                          stock_percentage: float, decision_type: str = "reorder") -> Dict:
        """Generate LIME explanation"""
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not available")
        
        X_input = np.array([current_stock, avg_daily_sales, stock_percentage])
        
        def predict_fn(data_array):
            df_temp = pd.DataFrame(data_array, columns=self.feature_names)
            return self.model.predict(df_temp)

        exp = self.lime_explainer.explain_instance(
            X_input,
            predict_fn,
            num_features=len(self.feature_names),
            num_samples=1000
        )
        
        lime_contributions = dict(exp.as_list())
        
        contributions = {}
        for key, value in lime_contributions.items():
            for feat in self.feature_names:
                if feat in key:
                    contributions[feat] = float(value)
                    break
        
        prediction = self.predict(current_stock, avg_daily_sales)['predicted_days']
        
        decision_rationale = self._generate_decision_rationale(
            contributions, current_stock, avg_daily_sales, prediction, decision_type
        )
        
        explanations = []
        for feature, impact in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "increases" if impact > 0 else "decreases"
            explanations.append(
                f"{feature} {direction} urgency ({impact:.2f})"
            )
        
        explanation_text = "; ".join(explanations)
        
        return {
            'explanation_type': 'LIME',
            'feature_contributions': contributions,
            'base_value': float(prediction - sum(contributions.values())),
            'prediction_value': float(prediction),
            'explanation_text': explanation_text,
            'decision_rationale': decision_rationale,
            'decision_type': decision_type
        }
    
    def _generate_decision_rationale(self, contributions: Dict, current_stock: float,
                                    avg_daily_sales: float, predicted_days: float,
                                    decision_type: str) -> str:
        """Generate human-readable rationale"""
        if decision_type == "reorder":
            if predicted_days < 3:
                return f"CRITICAL: Only {current_stock} units left. At {avg_daily_sales:.1f}/day, stockout in {predicted_days:.1f} days."
            elif predicted_days < 7:
                return f"URGENT: {current_stock} units remaining. Depletion predicted in {predicted_days:.1f} days."
            else:
                return f"STABLE: Sufficient stock ({current_stock}) for {predicted_days:.1f} days."
        return "Standard monitoring cycle."


# Initialize service
ml_service = EnhancedInventoryMLService()


@app.on_event("startup")
async def startup_event():
    """Initialize ML service on startup"""
    logger.info("🚀 Starting Enhanced ML + XAI Service...")
    
    if not ml_service.load_model():
        logger.info("No pre-trained model found - will train on first request")


@app.post("/train")
async def train_model():
    """Train/retrain the model"""
    success = ml_service.train_model()
    if success:
        return {"status": "success", "message": "Model trained successfully"}
    else:
        raise HTTPException(status_code=500, detail="Training failed")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict days to depletion"""
    if ml_service.model is None:
        ml_service.train_model()
    
    try:
        result = ml_service.predict(request.current_stock, request.avg_daily_sales)
        return PredictionResponse(
            product=request.product,
            predicted_days_to_depletion=result['predicted_days'],
            confidence=result['confidence'],
            recommendation=result['recommendation'],
            urgency_level=result['urgency_level']
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/shap", response_model=ExplanationResponse)
async def explain_shap(request: SHAPExplanationRequest):
    """Generate SHAP explanation"""
    if ml_service.shap_explainer is None:
        ml_service.train_model()
    
    try:
        # Get real data from CSV to ensure accuracy
        df = pd.read_csv('output/product_inventory_usage.csv')
        product_row = df[df['Product'] == request.product]
        
        if product_row.empty:
            # Fallback if product not yet sold/logged
            current_stock = 100
            avg_sales = 5.0
            pct = 100.0
        else:
            row = product_row.iloc[0]
            # Use columns if available, else infer
            current_stock = float(row.get('CurrentStock', 100))
            total_sold = float(row.get('TotalUnitsSold', 0))
            # Approximate daily sales if not stored directly
            avg_sales = total_sold / 5.0 if total_sold > 0 else 1.0
            
            starting = current_stock + total_sold
            pct = (current_stock / starting * 100) if starting > 0 else 0
            
        explanation = ml_service.explain_with_shap(current_stock, avg_sales, pct)
        
        return ExplanationResponse(
            product=request.product,
            explanation_type="SHAP",
            feature_contributions=explanation['feature_contributions'],
            base_value=explanation['base_value'],
            prediction_value=explanation['prediction_value'],
            explanation_text=explanation['explanation_text']
        )
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/lime", response_model=ExplanationResponse)
async def explain_lime(request: LIMEExplanationRequest):
    """Generate LIME explanation"""
    if ml_service.lime_explainer is None:
        ml_service.train_model()
    
    try:
        explanation = ml_service.explain_with_lime(
            request.current_stock, 
            request.avg_daily_sales, 
            request.stock_percentage, 
            request.decision_type
        )
        
        return ExplanationResponse(
            product=request.product,
            explanation_type="LIME",
            feature_contributions=explanation['feature_contributions'],
            base_value=explanation['base_value'],
            prediction_value=explanation['prediction_value'],
            explanation_text=explanation['explanation_text'],
            decision_rationale=explanation['decision_rationale']
        )
    except Exception as e:
        logger.error(f"LIME error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict():
    """
    Run predictions for all products using REAL inventory.
    INCLUDES AUTO-RETRAINING: Re-learns patterns every 60 seconds.
    """
    # ---------------------------------------------------------
    # 1. AUTO-RETRAIN LOGIC (The "Self-Healing" Brain)
    # ---------------------------------------------------------
    last_trained_str = ml_service.metadata.get('trained_at')
    should_retrain = False
    
    if not last_trained_str or ml_service.model is None:
        should_retrain = True
    else:
        # Check if 60 seconds have passed since last training
        last_trained = datetime.fromisoformat(last_trained_str)
        seconds_elapsed = (datetime.now() - last_trained).total_seconds()
        
        if seconds_elapsed > 60: 
            should_retrain = True
            logger.info(f"🔄 Model is {seconds_elapsed:.0f}s old - Retraining on latest data...")

    # Execute training if needed (Takes ~2 seconds)
    if should_retrain:
        success = ml_service.train_model()
        if not success:
            logger.warning("⚠️ Retraining failed, sticking to existing model")

    # ---------------------------------------------------------
    # 2. PREDICTION LOGIC (Using REAL Kafka Data)
    # ---------------------------------------------------------
    try:
        # Read the file updated by processor_consumer.py
        # This contains the ACTUAL stock count from the Kafka stream
        if not os.path.exists('output/product_inventory_usage.csv'):
            return {"status": "waiting", "message": "No inventory data found yet"}

        df = pd.read_csv('output/product_inventory_usage.csv')
        predictions = []
        
        for _, row in df.iterrows():
            product = row['Product']
            
            # [CRITICAL] USE REAL DATA - Do not default to 10,000!
            # We use .get() with a fallback of 0 to catch empty rows
            current_stock = float(row.get('CurrentStock', 0)) 
            total_sold = float(row.get('TotalUnitsSold', 0))
            
            # Calculate Sales Velocity (Avoid division by zero)
            # If total_sold is 0, we assume a tiny minimal sales rate (0.1) to avoid errors
            avg_daily_sales = max(0.1, total_sold / 5.0)
            
            # Generate Prediction using the (possibly new) model
            result = ml_service.predict(current_stock, avg_daily_sales)
            
            predictions.append({
                'Product': product,
                'CurrentStock': current_stock,
                'AvgDailySales': avg_daily_sales,
                'PredictedDaysToDepletion': result['predicted_days'],
                'Confidence': result['confidence'],
                'Recommendation': result['recommendation'],
                'UrgencyLevel': result['urgency_level'],
                'Timestamp': datetime.now().isoformat()
            })
        
        # Save predictions for the Agents/Dashboard to see
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv('output/predictions.csv', index=False)
        
        return {
            "status": "success",
            "products_predicted": len(predictions),
            "output_file": "output/predictions.csv",
            "model_retrained": should_retrain,
            "model_age_seconds": 0 if should_retrain else seconds_elapsed
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run ML service"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()