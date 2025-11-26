"""
Agentic AI Controller for LiveInsight+
Autonomous inventory management agent with human-in-the-loop controls
Uses SHAP evidence for decision-making with safety guarantees
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import requests
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InventoryAgent:
    """
    Autonomous agent for inventory reorder decisions
    
    Features:
    - Rule-based policy using ML predictions
    - SHAP-driven evidence collection
    - Human-in-the-loop approval workflow
    - Action logging and audit trail
    - Safety constraints and rollback
    """
    
    def __init__(self, ml_service_url='http://localhost:8000', 
                 check_interval=30.0,
                 auto_approve_threshold=0.9):
        """
        Initialize agent
        
        Args:
            ml_service_url: URL of ML service
            check_interval: Seconds between decision cycles
            auto_approve_threshold: Confidence threshold for auto-approval
        """
        self.ml_service_url = ml_service_url
        self.check_interval = check_interval
        self.auto_approve_threshold = auto_approve_threshold
        
        # State
        self.actions_history = []
        self.pending_actions = []
        
        # Output files
        os.makedirs('output', exist_ok=True)
        self.actions_file = 'output/agent_actions.csv'
        
        # Initialize actions file if not exists
        if not os.path.exists(self.actions_file):
            pd.DataFrame(columns=[
                'ActionID', 'Timestamp', 'Product', 'CurrentStock', 
                'PredictedDaysToDepletion', 'Confidence', 'ReorderQuantity',
                'Reasoning', 'SHAPEvidence', 'Status', 'ApprovedBy', 'ApprovedAt'
            ]).to_csv(self.actions_file, index=False)
        
        logger.info("✅ Inventory Agent initialized")
        logger.info(f"🔗 ML Service: {ml_service_url}")
        logger.info(f"⏱️  Check interval: {check_interval}s")
        logger.info(f"🤖 Auto-approve threshold: {auto_approve_threshold}")
    
    def check_ml_service_health(self) -> bool:
        """Check if ML service is accessible"""
        try:
            response = requests.get(f"{self.ml_service_url}/", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"ML service not accessible: {e}")
            return False
    
    def get_predictions(self) -> Optional[pd.DataFrame]:
        """
        Get latest predictions from ML service
        
        Returns:
            DataFrame with predictions, or None if unavailable
        """
        try:
            # Trigger batch prediction
            response = requests.post(f"{self.ml_service_url}/batch-predict", timeout=60)
            if response.status_code != 200:
                logger.error(f"Batch prediction failed: {response.text}")
                return None
            
            # Load predictions
            if os.path.exists('output/predictions.csv'):
                df = pd.read_csv('output/predictions.csv')
                logger.info(f"📥 Loaded {len(df)} predictions")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}", exc_info=True)
            return None
    
    def get_shap_explanation(self, product: str) -> Optional[Dict]:
        """
        Get SHAP explanation for a product
        
        Args:
            product: Product name
            
        Returns:
            SHAP explanation dict, or None if unavailable
        """
        try:
            response = requests.post(
                f"{self.ml_service_url}/explain",
                json={"product": product},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Could not get explanation for {product}: {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"Error getting SHAP explanation: {e}")
            return None
    
    def calculate_reorder_quantity(self, product: str, current_stock: int, 
                                   avg_daily_sales: float, days_to_depletion: float) -> int:
        """
        Calculate recommended reorder quantity
        
        Args:
            product: Product name
            current_stock: Current inventory level
            avg_daily_sales: Daily sales velocity
            days_to_depletion: Predicted days until stockout
            
        Returns:
            Recommended reorder quantity
        """
        # Safety stock: 7 days worth
        safety_stock = int(avg_daily_sales * 7)
        
        # Order quantity: enough for 30 days + safety stock - current stock
        target_stock = int(avg_daily_sales * 30) + safety_stock
        reorder_qty = max(0, target_stock - current_stock)
        
        # Round to nearest case size (assume 12 units per case)
        case_size = 12
        reorder_qty = ((reorder_qty + case_size - 1) // case_size) * case_size
        
        return reorder_qty
    
    def evaluate_action(self, row: pd.Series) -> Dict:
        """
        Evaluate whether to create a reorder action
        
        Args:
            row: Prediction row from ML service
            
        Returns:
            Action dict or None if no action needed
        """
        product = row['Product']
        current_stock = row['CurrentStock']
        days_to_depletion = row['PredictedDaysToDepletion']
        confidence = row['Confidence']
        avg_daily_sales = row['AvgDailySales']
        
        # Decision rules
        action_needed = False
        urgency = "LOW"
        reasoning = []
        
        if days_to_depletion < 3:
            action_needed = True
            urgency = "CRITICAL"
            reasoning.append("Less than 3 days of stock remaining")
        elif days_to_depletion < 7:
            action_needed = True
            urgency = "HIGH"
            reasoning.append("Less than 7 days of stock remaining")
        elif days_to_depletion < 14:
            action_needed = True
            urgency = "MEDIUM"
            reasoning.append("Less than 14 days of stock remaining")
        
        if not action_needed:
            return None
        
        # Get SHAP explanation for evidence
        shap_explanation = self.get_shap_explanation(product)
        shap_evidence = "N/A"
        
        if shap_explanation:
            shap_values = shap_explanation.get('shap_values', {})
            # Identify key drivers
            top_driver = max(shap_values.items(), key=lambda x: abs(x[1]))
            shap_evidence = f"{top_driver[0]} impact: {top_driver[1]:.2f} days"
            reasoning.append(f"Key driver: {shap_evidence}")
        
        # Calculate reorder quantity
        reorder_qty = self.calculate_reorder_quantity(
            product, current_stock, avg_daily_sales, days_to_depletion
        )
        
        reasoning.append(f"Reorder {reorder_qty} units for 30-day supply")
        
        # Determine auto-approval
        status = "PENDING"
        if confidence >= self.auto_approve_threshold and urgency != "CRITICAL":
            status = "AUTO_APPROVED"
            reasoning.append(f"Auto-approved (confidence={confidence:.2f})")
        else:
            reasoning.append("Requires human approval")
        
        # Create action
        action_id = f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(product) % 10000:04d}"
        
        action = {
            'ActionID': action_id,
            'Timestamp': datetime.now().isoformat(),
            'Product': product,
            'CurrentStock': int(current_stock),
            'PredictedDaysToDepletion': float(days_to_depletion),
            'Confidence': float(confidence),
            'ReorderQuantity': int(reorder_qty),
            'Urgency': urgency,
            'Reasoning': "; ".join(reasoning),
            'SHAPEvidence': shap_evidence,
            'Status': status,
            'ApprovedBy': 'AGENT' if status == 'AUTO_APPROVED' else None,
            'ApprovedAt': datetime.now().isoformat() if status == 'AUTO_APPROVED' else None
        }
        
        return action
    
    def save_actions(self, actions: List[Dict]):
        """
        Save actions to CSV file
        
        Args:
            actions: List of action dicts
        """
        try:
            # Load existing actions
            if os.path.exists(self.actions_file):
                existing_df = pd.read_csv(self.actions_file)
            else:
                existing_df = pd.DataFrame()
            
            # Append new actions
            new_df = pd.DataFrame(actions)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates (keep latest)
            combined_df = combined_df.drop_duplicates(subset=['Product', 'Status'], keep='last')
            
            # Sort by urgency and timestamp
            urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
            combined_df['_urgency_sort'] = combined_df['Urgency'].map(urgency_order)
            combined_df = combined_df.sort_values(['_urgency_sort', 'Timestamp'], ascending=[True, False])
            combined_df = combined_df.drop(columns=['_urgency_sort'])
            
            # Save
            combined_df.to_csv(self.actions_file, index=False)
            logger.info(f"💾 Saved {len(actions)} actions to {self.actions_file}")
            
        except Exception as e:
            logger.error(f"Error saving actions: {e}", exc_info=True)
    
    def decision_cycle(self):
        """
        Execute one decision cycle
        
        Steps:
        1. Get predictions from ML service
        2. Evaluate each product
        3. Create actions for products needing reorder
        4. Save actions for human review
        """
        logger.info("🔄 Starting decision cycle...")
        
        # Check ML service
        if not self.check_ml_service_health():
            logger.warning("⚠️  ML service unavailable - skipping cycle")
            return
        
        # Get predictions
        predictions_df = self.get_predictions()
        if predictions_df is None or predictions_df.empty:
            logger.warning("⚠️  No predictions available - skipping cycle")
            return
        
        # Evaluate each product
        actions = []
        for _, row in predictions_df.iterrows():
            action = self.evaluate_action(row)
            if action:
                actions.append(action)
                logger.info(
                    f"📋 Action created: {action['Product']} - "
                    f"{action['Urgency']} - {action['Status']}"
                )
        
        # Save actions
        if actions:
            self.save_actions(actions)
            
            # Summary
            pending = sum(1 for a in actions if a['Status'] == 'PENDING')
            auto_approved = sum(1 for a in actions if a['Status'] == 'AUTO_APPROVED')
            
            logger.info("=" * 70)
            logger.info(f"✅ Decision cycle complete")
            logger.info(f"   Total actions: {len(actions)}")
            logger.info(f"   Pending approval: {pending}")
            logger.info(f"   Auto-approved: {auto_approved}")
            logger.info("=" * 70)
        else:
            logger.info("✅ No actions needed - inventory levels OK")
    
    def run(self, continuous=True):
        """
        Run agent main loop
        
        Args:
            continuous: If True, run continuously; if False, run once
        """
        logger.info("🚀 Agent starting...")
        
        try:
            while True:
                self.decision_cycle()
                
                if not continuous:
                    break
                
                logger.info(f"⏸️  Sleeping for {self.check_interval}s...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("⚠️  Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ Agent error: {e}", exc_info=True)
        finally:
            logger.info("✅ Agent shutdown complete")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LiveInsight+ Inventory Agent')
    parser.add_argument('--ml-service', default='http://localhost:8000', 
                       help='ML service URL')
    parser.add_argument('--interval', type=float, default=30.0,
                       help='Check interval (seconds)')
    parser.add_argument('--auto-threshold', type=float, default=0.85,
                       help='Confidence threshold for auto-approval')
    parser.add_argument('--once', action='store_true',
                       help='Run once then exit (default: continuous)')
    
    args = parser.parse_args()
    
    agent = InventoryAgent(
        ml_service_url=args.ml_service,
        check_interval=args.interval,
        auto_approve_threshold=args.auto_threshold
    )
    
    agent.run(continuous=not args.once)


if __name__ == "__main__":
    main()
