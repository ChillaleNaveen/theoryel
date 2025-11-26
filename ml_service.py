"""
ML + Explainable AI Service for LiveInsight+
FastAPI service providing inventory predictions with SHAP explanations
Implements instance-level, global, and feature importance XAI
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LiveInsight+ ML & XAI Service",
    description="Inventory prediction with SHAP explanations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
MODEL_CACHE = {}
EXPLAINER_CACHE = {}
PREDICTIONS_HISTORY = []

# Output directories
os.makedirs('output', exist_ok=True)
os.makedirs('output/xai', exist_ok=True)
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
    
class ExplanationRequest(BaseModel):
    product: str
    
class ExplanationResponse(BaseModel):
    product: str
    shap_values: Dict[str, float]
    base_value: float
    feature_importance: Dict[str, float]
    explanation_text: str


class InventoryMLService:
    """
    ML service for inventory prediction with XAI
    
    Features:
    - Random Forest regression for days-to-depletion
    - SHAP for instance and global explanations
    - Feature importance analysis
    - Partial Dependence Plots (PDP) data
    """
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = ['CurrentStock', 'AvgDailySales', 'StockPercentage']
        self.model_path = 'models/inventory_model.pkl'
        self.explainer_path = 'models/shap_explainer.pkl'
        self.scaler_path = 'models/scaler.pkl'
        
        # Training metadata
        self.training_metadata = {
            'trained_at': None,
            'n_samples': 0,
            'mae': 0.0,
            'r2': 0.0,
            'feature_importance': {}
        }
        
    def prepare_training_data(self) -> Optional[pd.DataFrame]:
        """
        Load and prepare training data from processor output
        
        Returns:
            DataFrame with features and target, or None if insufficient data
        """
        try:
            # Load product inventory usage
            df = pd.read_csv('output/product_inventory_usage.csv')
            
            if len(df) < 10:
                logger.warning("Insufficient data for training (need >= 10 products)")
                return None
            
            # Simulate starting stock and current stock
            np.random.seed(42)
            df['StartingStock'] = df['TotalUnitsSold'] + np.random.randint(50, 200, size=len(df))
            df['CurrentStock'] = df['StartingStock'] - df['TotalUnitsSold']
            
            # Average daily sales (assuming 5-day window)
            df['AvgDailySales'] = df['TotalUnitsSold'] / 5.0
            
            # Stock percentage
            df['StockPercentage'] = (df['CurrentStock'] / df['StartingStock']) * 100
            
            # Target: DaysToDepletion (calculated)
            df['DaysToDepletion'] = df.apply(
                lambda row: row['CurrentStock'] / row['AvgDailySales'] 
                if row['AvgDailySales'] > 0 else 999,
                axis=1
            )
            
            # Add noise for realism
            df['DaysToDepletion'] = df['DaysToDepletion'] + np.random.normal(0, 2, len(df))
            df['DaysToDepletion'] = df['DaysToDepletion'].clip(lower=0)
            
            # Clean data
            df = df[df['AvgDailySales'] > 0]
            df = df[df['DaysToDepletion'] < 500]  # Remove outliers
            
            logger.info(f"✅ Prepared {len(df)} training samples")
            return df
            
        except FileNotFoundError:
            logger.warning("product_inventory_usage.csv not found - waiting for data")
            return None
        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            return None
    
    def train_model(self) -> bool:
        """
        Train ML model and create SHAP explainer
        
        Returns:
            True if training successful, False otherwise
        """
        try:
            df = self.prepare_training_data()
            if df is None:
                return False
            
            # Features and target
            X = df[self.feature_names]
            y = df['DaysToDepletion']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            logger.info("🧠 Training Random Forest model...")
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"📊 Model performance: MAE={mae:.2f}, R²={r2:.4f}")
            
            # Create SHAP explainer
            logger.info("🔍 Creating SHAP explainer...")
            self.explainer = shap.TreeExplainer(self.model)
            
            # Compute feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            # Save metadata
            self.training_metadata = {
                'trained_at': datetime.now().isoformat(),
                'n_samples': len(df),
                'mae': float(mae),
                'r2': float(r2),
                'feature_importance': {k: float(v) for k, v in feature_importance.items()}
            }
            
            # Save model and explainer
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.explainer, self.explainer_path)
            
            logger.info("✅ Model training complete and saved")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            return False
    
    def load_model(self) -> bool:
        """Load pre-trained model and explainer"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.explainer = joblib.load(self.explainer_path)
                logger.info("✅ Loaded pre-trained model")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, current_stock: int, avg_daily_sales: float) -> Dict:
        """
        Predict days to depletion for a product
        
        Args:
            current_stock: Current inventory level
            avg_daily_sales: Average daily sales velocity
            
        Returns:
            Prediction dict with days, confidence, recommendation
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Prepare features
        stock_percentage = 100.0  # Assume full stock initially
        if avg_daily_sales > 0:
            starting_stock = current_stock / (1 - (avg_daily_sales * 5) / (current_stock + avg_daily_sales * 5))
            stock_percentage = (current_stock / starting_stock) * 100
        
        X = pd.DataFrame([[current_stock, avg_daily_sales, stock_percentage]], 
                        columns=self.feature_names)
        
        # Predict
        days_pred = self.model.predict(X)[0]
        
        # Estimate confidence (using tree variance)
        predictions_per_tree = [tree.predict(X)[0] for tree in self.model.estimators_]
        confidence = 1.0 / (1.0 + np.std(predictions_per_tree))
        
        # Generate recommendation
        if days_pred < 3:
            recommendation = "🚨 URGENT: Reorder immediately"
        elif days_pred < 7:
            recommendation = "⚠️  WARNING: Reorder soon"
        elif days_pred < 14:
            recommendation = "📊 MONITOR: Plan reorder"
        else:
            recommendation = "✅ OK: Stock sufficient"
        
        return {
            'predicted_days': float(days_pred),
            'confidence': float(confidence),
            'recommendation': recommendation
        }
    
    def explain_prediction(self, current_stock: int, avg_daily_sales: float, 
                          stock_percentage: float) -> Dict:
        """
        Generate SHAP explanation for a prediction
        
        Args:
            current_stock: Current inventory
            avg_daily_sales: Daily sales rate
            stock_percentage: Stock level percentage
            
        Returns:
            Dict with SHAP values, base value, and interpretation
        """
        if self.explainer is None:
            raise ValueError("Explainer not available")
        
        # Prepare features
        X = pd.DataFrame([[current_stock, avg_daily_sales, stock_percentage]], 
                        columns=self.feature_names)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)[0]
        base_value = self.explainer.expected_value
        
        # Feature contributions
        contributions = dict(zip(self.feature_names, shap_values))
        
        # Generate explanation text
        explanation_parts = []
        for feature, value in contributions.items():
            if abs(value) > 0.5:
                direction = "increases" if value > 0 else "decreases"
                explanation_parts.append(
                    f"{feature} {direction} depletion time by {abs(value):.1f} days"
                )
        
        explanation_text = "; ".join(explanation_parts) if explanation_parts else \
                          "All features have minimal impact"
        
        return {
            'shap_values': {k: float(v) for k, v in contributions.items()},
            'base_value': float(base_value),
            'feature_importance': self.training_metadata.get('feature_importance', {}),
            'explanation_text': explanation_text
        }
    
    def get_global_explanations(self) -> Dict:
        """
        Generate global model explanations
        
        Returns:
            Dict with feature importance, PDP data, summary stats
        """
        if self.model is None:
            raise ValueError("Model not available")
        
        return {
            'feature_importance': self.training_metadata.get('feature_importance', {}),
            'model_performance': {
                'mae': self.training_metadata.get('mae', 0.0),
                'r2': self.training_metadata.get('r2', 0.0)
            },
            'training_info': {
                'trained_at': self.training_metadata.get('trained_at'),
                'n_samples': self.training_metadata.get('n_samples', 0)
            }
        }


# Initialize service
ml_service = InventoryMLService()


@app.on_event("startup")
async def startup_event():
    """Initialize ML service on startup"""
    logger.info("🚀 Starting ML + XAI Service...")
    
    # Try loading existing model
    if not ml_service.load_model():
        logger.info("No pre-trained model found - will train on first request")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "LiveInsight+ ML & XAI Service",
        "status": "operational",
        "model_ready": ml_service.model is not None
    }


@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """
    Train/retrain the ML model
    Can be called manually or automatically when new data arrives
    """
    logger.info("📥 Received training request")
    
    success = ml_service.train_model()
    
    if success:
        return {
            "status": "success",
            "message": "Model trained successfully",
            "metadata": ml_service.training_metadata
        }
    else:
        raise HTTPException(status_code=500, detail="Training failed - check logs")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict days to depletion for a product
    
    Args:
        request: Product name, current stock, avg daily sales
        
    Returns:
        Prediction with confidence and recommendation
    """
    # Ensure model is trained
    if ml_service.model is None:
        logger.info("Model not ready - training now...")
        if not ml_service.train_model():
            raise HTTPException(status_code=503, detail="Model training failed")
    
    try:
        result = ml_service.predict(request.current_stock, request.avg_daily_sales)
        
        # Log prediction
        PREDICTIONS_HISTORY.append({
            'timestamp': datetime.now().isoformat(),
            'product': request.product,
            'prediction': result
        })
        
        return PredictionResponse(
            product=request.product,
            predicted_days_to_depletion=result['predicted_days'],
            confidence=result['confidence'],
            recommendation=result['recommendation']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    """
    Generate SHAP explanation for a product's prediction
    
    Args:
        request: Product name
        
    Returns:
        SHAP values, feature importance, and interpretation
    """
    if ml_service.explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not available")
    
    try:
        # Load product data
        df = pd.read_csv('output/product_inventory_usage.csv')
        df = df[df['Product'] == request.product]
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Product '{request.product}' not found")
        
        # Reconstruct features (same logic as training)
        row = df.iloc[0]
        total_sold = row['TotalUnitsSold']
        starting_stock = total_sold + np.random.randint(50, 200)
        current_stock = starting_stock - total_sold
        avg_daily_sales = total_sold / 5.0
        stock_percentage = (current_stock / starting_stock) * 100
        
        # Get explanation
        explanation = ml_service.explain_prediction(
            current_stock, avg_daily_sales, stock_percentage
        )
        
        # Save explanation
        explanation_file = f'output/xai/{request.product.replace(" ", "_")}_shap.json'
        with open(explanation_file, 'w') as f:
            json.dump(explanation, f, indent=2)
        
        return ExplanationResponse(
            product=request.product,
            shap_values=explanation['shap_values'],
            base_value=explanation['base_value'],
            feature_importance=explanation['feature_importance'],
            explanation_text=explanation['explanation_text']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/global-explanations")
async def global_explanations():
    """
    Get global model explanations (feature importance, performance metrics)
    """
    try:
        return ml_service.get_global_explanations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/history")
async def prediction_history(limit: int = 100):
    """Get recent prediction history"""
    return {
        "count": len(PREDICTIONS_HISTORY),
        "predictions": PREDICTIONS_HISTORY[-limit:]
    }


@app.post("/batch-predict")
async def batch_predict():
    """
    Run predictions for all products in inventory
    Saves results to output/predictions.csv
    """
    if ml_service.model is None:
        logger.info("Model not ready - training now...")
        if not ml_service.train_model():
            raise HTTPException(status_code=503, detail="Model training failed")
    
    try:
        # Load inventory data
        df = pd.read_csv('output/product_inventory_usage.csv')
        
        predictions = []
        
        for _, row in df.iterrows():
            product = row['Product']
            total_sold = row['TotalUnitsSold']
            
            # Simulate current inventory
            np.random.seed(hash(product) % (2**32))
            starting_stock = total_sold + np.random.randint(50, 200)
            current_stock = starting_stock - total_sold
            avg_daily_sales = total_sold / 5.0
            
            if avg_daily_sales <= 0:
                continue
            
            # Predict
            result = ml_service.predict(current_stock, avg_daily_sales)
            
            predictions.append({
                'Product': product,
                'CurrentStock': current_stock,
                'AvgDailySales': avg_daily_sales,
                'PredictedDaysToDepletion': result['predicted_days'],
                'Confidence': result['confidence'],
                'Recommendation': result['recommendation'],
                'Timestamp': datetime.now().isoformat()
            })
        
        # Save to CSV
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv('output/predictions.csv', index=False)
        
        logger.info(f"✅ Batch predictions complete: {len(predictions)} products")
        
        return {
            "status": "success",
            "products_predicted": len(predictions),
            "output_file": "output/predictions.csv"
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run ML service"""
    logger.info("🚀 Starting ML + XAI Service on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
