"""
Enhanced Agentic AI Controller with LIME Explanations
Autonomous inventory management agent that uses LIME for human-interpretable decisions
Provides transparent, actionable justifications for every reorder recommendation
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import requests
import json
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LIMEPoweredInventoryAgent:
    """
    Autonomous agent with LIME-based transparent decision-making
    
    Features:
    - LIME explanations for every reorder decision
    - Human-readable justifications
    - Evidence-based urgency classification
    - Audit trail with decision rationale
    - Safety constraints and human-in-the-loop
    """
    
    def __init__(self, ml_service_url='http://localhost:8000', 
                 check_interval=30.0,
                 auto_approve_threshold=0.85):
        """
        Initialize LIME-powered agent
        
        Args:
            ml_service_url: URL of enhanced ML service
            check_interval: Seconds between decision cycles
            auto_approve_threshold: Confidence threshold for auto-approval
        """
        self.ml_service_url = ml_service_url
        self.check_interval = check_interval
        self.auto_approve_threshold = auto_approve_threshold
        
        # State
        self.actions_history = []
        self.lime_explanations_cache = {}
        
        # Output files
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/agent_lime_decisions', exist_ok=True)
        self.actions_file = 'output/agent_actions_with_lime.csv'
        self.metrics_file = 'output/agent_performance_metrics.csv'
        
        # Initialize files
        if not os.path.exists(self.actions_file):
            pd.DataFrame(columns=[
                'ActionID', 'Timestamp', 'Product', 'CurrentStock', 
                'PredictedDaysToDepletion', 'Confidence', 'UrgencyLevel',
                'ReorderQuantity', 'Reasoning', 'LIMEEvidence', 
                'DecisionRationale', 'Status', 'ApprovedBy', 'ApprovedAt',
                'ExplanationType'
            ]).to_csv(self.actions_file, index=False)
        
        # Initialize metrics tracking
        if not os.path.exists(self.metrics_file):
            pd.DataFrame(columns=[
                'Timestamp', 'CycleID', 'ProductsAnalyzed', 'ActionsCreated',
                'CriticalActions', 'HighActions', 'MediumActions',
                'AutoApproved', 'PendingApproval', 'ProcessingTimeSeconds',
                'AverageLIMEExplanationTime'
            ]).to_csv(self.metrics_file, index=False)
        
        logger.info("✅ LIME-Powered Inventory Agent initialized")
        logger.info(f"🔗 ML Service: {ml_service_url}")
        logger.info(f"⏱️  Check interval: {check_interval}s")
        logger.info(f"🤖 Auto-approve threshold: {auto_approve_threshold}")
        logger.info(f"🔍 Explanation method: LIME (Local Interpretable Model-agnostic Explanations)")
    
    def check_ml_service_health(self) -> bool:
        """Check if enhanced ML service is accessible"""
        try:
            response = requests.get(f"{self.ml_service_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                xai_ready = data.get('xai_ready', {})
                if not xai_ready.get('lime', False):
                    logger.warning("⚠️  LIME explainer not ready in ML service")
                    return False
                return True
            return False
        except Exception as e:
            logger.warning(f"ML service not accessible: {e}")
            return False
    
    def get_predictions(self) -> Optional[pd.DataFrame]:
        """Get latest predictions from ML service"""
        try:
            response = requests.post(f"{self.ml_service_url}/batch-predict", timeout=60)
            if response.status_code != 200:
                logger.error(f"Batch prediction failed: {response.text}")
                return None
            
            if os.path.exists('output/predictions.csv'):
                df = pd.read_csv('output/predictions.csv')
                logger.info(f"📥 Loaded {len(df)} predictions")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}", exc_info=True)
            return None
    
    def get_lime_explanation(self, product: str, decision_type: str = "reorder") -> Optional[Dict]:
        """
        Get LIME explanation for a product decision
        
        Args:
            product: Product name
            decision_type: Type of decision (reorder, urgency, quantity)
            
        Returns:
            LIME explanation dict with decision rationale
        """
        cache_key = f"{product}_{decision_type}"
        
        # Check cache first
        if cache_key in self.lime_explanations_cache:
            return self.lime_explanations_cache[cache_key]
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.ml_service_url}/explain/lime",
                json={
                    "product": product,
                    "decision_type": decision_type
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                explanation = response.json()
                self.lime_explanations_cache[cache_key] = explanation
                logger.debug(f"✅ LIME explanation for {product} ({decision_type}) in {elapsed:.2f}s")
                return explanation
            else:
                logger.warning(f"Could not get LIME explanation for {product}: {response.text}")
                return None
                
        except Exception as e:
            logger.warning(f"Error getting LIME explanation: {e}")
            return None
    
    def calculate_reorder_quantity(self, current_stock: int, 
                                   avg_daily_sales: float, 
                                   days_to_depletion: float,
                                   urgency_level: str) -> int:
        """
        Calculate recommended reorder quantity based on urgency
        """
        # Base parameters
        safety_days = 7
        
        # Adjust target based on urgency
        if urgency_level == "CRITICAL":
            target_days = 45  # Order more for critical items
        elif urgency_level == "HIGH":
            target_days = 35
        elif urgency_level == "MEDIUM":
            target_days = 30
        else:
            target_days = 25
        
        # Calculate quantities
        safety_stock = int(avg_daily_sales * safety_days)
        target_stock = int(avg_daily_sales * target_days)
        reorder_qty = max(0, target_stock + safety_stock - current_stock)
        
        # Round to case size (12 units per case)
        case_size = 12
        reorder_qty = ((reorder_qty + case_size - 1) // case_size) * case_size
        
        return reorder_qty
    
    def evaluate_action(self, row: pd.Series, lime_explanation_time: list) -> Optional[Dict]:
        """
        Evaluate whether to create a reorder action using LIME explanations
        
        Args:
            row: Prediction row from ML service
            lime_explanation_time: List to track LIME processing times
            
        Returns:
            Action dict with LIME rationale or None if no action needed
        """
        product = row['Product']
        current_stock = int(row['CurrentStock'])
        days_to_depletion = float(row['PredictedDaysToDepletion'])
        confidence = float(row['Confidence'])
        avg_daily_sales = float(row['AvgDailySales'])
        urgency_level = row['UrgencyLevel']
        
        # Decision rules
        action_needed = False
        reasoning = []
        
        if urgency_level == "CRITICAL":
            action_needed = True
            reasoning.append(f"CRITICAL urgency: {days_to_depletion:.1f} days to stockout")
        elif urgency_level == "HIGH":
            action_needed = True
            reasoning.append(f"HIGH urgency: {days_to_depletion:.1f} days to stockout")
        elif urgency_level == "MEDIUM":
            action_needed = True
            reasoning.append(f"MEDIUM urgency: {days_to_depletion:.1f} days remaining")
        
        if not action_needed:
            return None
        
        # Get LIME explanation for this decision
        lime_start = time.time()
        lime_explanation = self.get_lime_explanation(product, "reorder")
        lime_elapsed = time.time() - lime_start
        lime_explanation_time.append(lime_elapsed)
        
        # Extract LIME evidence and rationale
        lime_evidence = "N/A"
        decision_rationale = "No LIME explanation available"
        
        if lime_explanation:
            # Get feature contributions
            contributions = lime_explanation.get('feature_contributions', {})
            
            # Format LIME evidence
            top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
            lime_evidence = "; ".join([
                f"{feat}: {val:+.2f}" for feat, val in top_features
            ])
            
            # Get decision rationale (human-readable)
            decision_rationale = lime_explanation.get('decision_rationale', 
                                                     'LIME analysis complete')
            
            reasoning.append(f"LIME: {lime_explanation.get('explanation_text', 'See detailed analysis')}")
        else:
            reasoning.append("LIME explanation unavailable - using rule-based logic")
        
        # Calculate reorder quantity
        reorder_qty = self.calculate_reorder_quantity(
            current_stock, avg_daily_sales, days_to_depletion, urgency_level
        )
        
        reasoning.append(f"Recommended order: {reorder_qty} units for {30}-day coverage")
        
        # Determine auto-approval
        status = "PENDING"
        approved_by = None
        
        if confidence >= self.auto_approve_threshold and urgency_level not in ["CRITICAL"]:
            status = "AUTO_APPROVED"
            approved_by = "AGENT"
            reasoning.append(f"Auto-approved (confidence={confidence:.2f}, urgency={urgency_level})")
        else:
            reasoning.append(f"Requires human approval (confidence={confidence:.2f}, urgency={urgency_level})")
        
        # Create action with LIME evidence
        action_id = f"ACT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(product) % 10000:04d}"
        
        action = {
            'ActionID': action_id,
            'Timestamp': datetime.now().isoformat(),
            'Product': product,
            'CurrentStock': current_stock,
            'PredictedDaysToDepletion': days_to_depletion,
            'Confidence': confidence,
            'UrgencyLevel': urgency_level,
            'ReorderQuantity': reorder_qty,
            'Reasoning': "; ".join(reasoning),
            'LIMEEvidence': lime_evidence,
            'DecisionRationale': decision_rationale,
            'Status': status,
            'ApprovedBy': approved_by,
            'ApprovedAt': datetime.now().isoformat() if status == 'AUTO_APPROVED' else None,
            'ExplanationType': 'LIME'
        }
        
        return action
    
    def save_actions(self, actions: List[Dict]):
        """Save actions to CSV with LIME evidence"""
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
            combined_df = combined_df.drop_duplicates(
                subset=['Product', 'Status'], keep='last'
            )
            
            # Sort by urgency and timestamp
            urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'NORMAL': 4}
            combined_df['_urgency_sort'] = combined_df['UrgencyLevel'].map(urgency_order)
            combined_df = combined_df.sort_values(
                ['_urgency_sort', 'Timestamp'], ascending=[True, False]
            )
            combined_df = combined_df.drop(columns=['_urgency_sort'])
            
            # Save
            combined_df.to_csv(self.actions_file, index=False)
            logger.info(f"💾 Saved {len(actions)} actions with LIME evidence to {self.actions_file}")
            
            # Save individual LIME decision files for detailed review
            for action in actions:
                decision_file = f"output/agent_lime_decisions/{action['ActionID']}.json"
                with open(decision_file, 'w') as f:
                    json.dump(action, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving actions: {e}", exc_info=True)
    
    def save_metrics(self, metrics: Dict):
        """Save performance metrics"""
        try:
            if os.path.exists(self.metrics_file):
                existing_df = pd.read_csv(self.metrics_file)
            else:
                existing_df = pd.DataFrame()
            
            new_df = pd.DataFrame([metrics])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Keep only last 1000 entries
            if len(combined_df) > 1000:
                combined_df = combined_df.tail(1000)
            
            combined_df.to_csv(self.metrics_file, index=False)
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)
    
    def decision_cycle(self):
        """Execute one decision cycle with LIME explanations"""
        cycle_id = datetime.now().strftime('%Y%m%d%H%M%S')
        cycle_start = time.time()
        
        logger.info("=" * 70)
        logger.info(f"🔄 Decision Cycle {cycle_id} - LIME-Powered Agent")
        logger.info("=" * 70)
        
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
        lime_times = []
        
        for _, row in predictions_df.iterrows():
            action = self.evaluate_action(row, lime_times)
            if action:
                actions.append(action)
                logger.info(
                    f"📋 Action: {action['Product']} | "
                    f"Urgency: {action['UrgencyLevel']} | "
                    f"Status: {action['Status']} | "
                    f"LIME: ✓"
                )
        
        # Save actions
        if actions:
            self.save_actions(actions)
            
            # Calculate metrics
            cycle_time = time.time() - cycle_start
            metrics = {
                'Timestamp': datetime.now().isoformat(),
                'CycleID': cycle_id,
                'ProductsAnalyzed': len(predictions_df),
                'ActionsCreated': len(actions),
                'CriticalActions': sum(1 for a in actions if a['UrgencyLevel'] == 'CRITICAL'),
                'HighActions': sum(1 for a in actions if a['UrgencyLevel'] == 'HIGH'),
                'MediumActions': sum(1 for a in actions if a['UrgencyLevel'] == 'MEDIUM'),
                'AutoApproved': sum(1 for a in actions if a['Status'] == 'AUTO_APPROVED'),
                'PendingApproval': sum(1 for a in actions if a['Status'] == 'PENDING'),
                'ProcessingTimeSeconds': cycle_time,
                'AverageLIMEExplanationTime': np.mean(lime_times) if lime_times else 0.0
            }
            
            self.save_metrics(metrics)
            
            # Summary
            logger.info("=" * 70)
            logger.info(f"✅ Decision cycle complete | Duration: {cycle_time:.2f}s")
            logger.info(f"   Products analyzed: {metrics['ProductsAnalyzed']}")
            logger.info(f"   Actions created: {metrics['ActionsCreated']}")
            logger.info(f"   CRITICAL: {metrics['CriticalActions']} | "
                       f"HIGH: {metrics['HighActions']} | "
                       f"MEDIUM: {metrics['MediumActions']}")
            logger.info(f"   Auto-approved: {metrics['AutoApproved']} | "
                       f"Pending: {metrics['PendingApproval']}")
            logger.info(f"   Avg LIME time: {metrics['AverageLIMEExplanationTime']:.3f}s")
            logger.info("=" * 70)
        else:
            logger.info("✅ No actions needed - inventory levels OK")
    
    def run(self, continuous=True):
        """Run agent main loop"""
        logger.info("🚀 LIME-Powered Agent starting...")
        logger.info("🔍 All decisions will be explained using LIME")
        
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
    
    parser = argparse.ArgumentParser(description='LIME-Powered Inventory Agent')
    parser.add_argument('--ml-service', default='http://localhost:8000', 
                       help='ML service URL')
    parser.add_argument('--interval', type=float, default=30.0,
                       help='Check interval (seconds)')
    parser.add_argument('--auto-threshold', type=float, default=0.85,
                       help='Confidence threshold for auto-approval')
    parser.add_argument('--once', action='store_true',
                       help='Run once then exit (default: continuous)')
    
    args = parser.parse_args()
    
    agent = LIMEPoweredInventoryAgent(
        ml_service_url=args.ml_service,
        check_interval=args.interval,
        auto_approve_threshold=args.auto_threshold
    )
    
    agent.run(continuous=not args.once)


if __name__ == "__main__":
    main()
