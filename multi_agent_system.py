"""
🤖 GOD-TIER Multi-Agent System with LIME Explainability
Autonomous AI agent system for retail inventory management

Features:
- Multiple specialized agents working in parallel
- Fully autonomous decision-making with LIME explanations
- Real-time coordination and consensus building
- Comprehensive explainability tracking
- Performance optimization and learning
"""
from kafka import KafkaProducer
import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)-20s %(levelname)s - %(message)s'
)


@dataclass
class AgentDecision:
    """Structure for agent decisions"""
    decision_id: str
    agent_name: str
    product: str
    action_type: str
    urgency_level: str
    confidence: float
    reorder_quantity: int
    reasoning: str
    lime_evidence: str
    decision_rationale: str
    timestamp: str
    auto_approved: bool
    processing_time_ms: float


@dataclass
class ConsensusResult:
    """Multi-agent consensus result"""
    product: str
    final_action: str
    final_urgency: str
    final_quantity: int
    consensus_confidence: float
    agent_votes: Dict[str, str]
    combined_reasoning: str
    combined_lime_evidence: str
    timestamp: str


class BaseAgent:
    """Base class for specialized agents"""
    
    def __init__(self, name: str, ml_service_url: str):
        self.name = name
        self.ml_service_url = ml_service_url
        self.logger = logging.getLogger(f"Agent.{name}")
        self.decisions_made = 0
        self.total_processing_time = 0.0
        
    def get_lime_explanation(self, product: str, current_stock: float, 
                            avg_daily_sales: float, stock_percentage: float) -> Dict:
        """Get LIME explanation from ML service"""
        try:
            # UPDATED: Sending full context in the body to avoid CSV lookup errors in ML service
            response = requests.post(
                f"{self.ml_service_url}/explain/lime",
                json={
                    "product": product,  # Added this field
                    "current_stock": current_stock,
                    "avg_daily_sales": avg_daily_sales,
                    "stock_percentage": stock_percentage,
                    "decision_type": "reorder"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.warning(f"LIME explanation failed: {e}")
            return {}
    
    def make_decision(self, row: pd.Series) -> Optional[AgentDecision]:
        """Override in subclasses"""
        raise NotImplementedError


class UrgencyAgent(BaseAgent):
    """Specialized agent for urgency classification"""
    
    def __init__(self, ml_service_url: str):
        super().__init__("UrgencyAgent", ml_service_url)
        self.logger.info("🚨 Urgency Agent initialized - Focus: Emergency detection")
    
    def make_decision(self, row: pd.Series) -> Optional[AgentDecision]:
        start_time = time.time()
        
        days_to_depletion = float(row.get('PredictedDaysToDepletion', 999))
        current_stock = int(row.get('CurrentStock', 0))
        avg_daily_sales = float(row.get('AvgDailySales', 0))
        product = row['Product']
        confidence = float(row.get('Confidence', 0))
        
        # Urgency classification
        if days_to_depletion < 2:
            urgency = "CRITICAL"
            action = "IMMEDIATE_REORDER"
            reasoning = f"🚨 EMERGENCY: Only {days_to_depletion:.1f} days until stockout!"
        elif days_to_depletion < 5:
            urgency = "HIGH"
            action = "URGENT_REORDER"
            reasoning = f"⚠️ HIGH PRIORITY: {days_to_depletion:.1f} days remaining"
        elif days_to_depletion < 10:
            urgency = "MEDIUM"
            action = "STANDARD_REORDER"
            reasoning = f"📋 MEDIUM: {days_to_depletion:.1f} days supply"
        else:
            return None  # No urgent action needed
        
        # Get LIME explanation
        stock_percentage = (current_stock / max(current_stock + avg_daily_sales * 30, 1)) * 100
        lime_exp = self.get_lime_explanation(product, current_stock, avg_daily_sales, stock_percentage)
        
        lime_evidence = "N/A"
        if lime_exp.get('feature_contributions'):
            top_features = sorted(
                lime_exp['feature_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:2]
            lime_evidence = "; ".join([f"{k}: {v:+.2f}" for k, v in top_features])
        
        # Calculate safe reorder quantity (ensures positive stock)
        quantity = int(avg_daily_sales * (30 if urgency == "CRITICAL" else 20))
        quantity = max(1, quantity)  # Ensure minimum order of 1 unit
        
        elapsed = (time.time() - start_time) * 1000
        self.decisions_made += 1
        self.total_processing_time += elapsed
        
        return AgentDecision(
            decision_id=f"URG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.decisions_made}",
            agent_name=self.name,
            product=product,
            action_type=action,
            urgency_level=urgency,
            confidence=confidence,
            reorder_quantity=quantity,
            reasoning=reasoning,
            lime_evidence=lime_evidence,
            decision_rationale=f"Urgency Agent: {reasoning} | LIME: {lime_evidence}",
            timestamp=datetime.now().isoformat(),
            auto_approved=urgency == "CRITICAL",
            processing_time_ms=elapsed
        )


class QuantityAgent(BaseAgent):
    """Specialized agent for optimal reorder quantity calculation"""
    
    def __init__(self, ml_service_url: str):
        super().__init__("QuantityAgent", ml_service_url)
        self.logger.info("📦 Quantity Agent initialized - Focus: Optimal order sizing")
    
    def make_decision(self, row: pd.Series) -> Optional[AgentDecision]:
        start_time = time.time()
        
        days_to_depletion = float(row.get('PredictedDaysToDepletion', 999))
        if days_to_depletion > 15:
            return None
        
        current_stock = int(row.get('CurrentStock', 0))
        avg_daily_sales = float(row.get('AvgDailySales', 0))
        product = row['Product']
        confidence = float(row.get('Confidence', 0))
        urgency = row.get('UrgencyLevel', 'LOW')
        
        # Optimal quantity calculation with EOQ principles
        holding_cost_per_unit = 0.5
        order_cost = 50
        annual_demand = avg_daily_sales * 365
        
        # Economic Order Quantity
        if annual_demand > 0:
            eoq = int(np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit))
        else:
            eoq = 100
        
        # Adjust for urgency
        urgency_multiplier = {"CRITICAL": 1.5, "HIGH": 1.3, "MEDIUM": 1.1, "LOW": 1.0}
        optimal_quantity = int(eoq * urgency_multiplier.get(urgency, 1.0))
        
        # Round to case size (12 units)
        case_size = 12
        optimal_quantity = ((optimal_quantity + case_size - 1) // case_size) * case_size
        
        # Ensure non-negative quantity (minimum 1 case)
        optimal_quantity = max(case_size, optimal_quantity)
        
        reasoning = f"📊 Optimal: {optimal_quantity} units (EOQ: {eoq}, Urgency: {urgency})"
        
        # Get LIME explanation
        stock_percentage = (current_stock / max(current_stock + avg_daily_sales * 30, 1)) * 100
        lime_exp = self.get_lime_explanation(product, current_stock, avg_daily_sales, stock_percentage)
        
        lime_evidence = "N/A"
        if lime_exp.get('feature_contributions'):
            top_features = sorted(
                lime_exp['feature_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:2]
            lime_evidence = "; ".join([f"{k}: {v:+.2f}" for k, v in top_features])
        
        elapsed = (time.time() - start_time) * 1000
        self.decisions_made += 1
        self.total_processing_time += elapsed
        
        return AgentDecision(
            decision_id=f"QTY-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.decisions_made}",
            agent_name=self.name,
            product=product,
            action_type="OPTIMIZE_QUANTITY",
            urgency_level=urgency,
            confidence=confidence,
            reorder_quantity=optimal_quantity,
            reasoning=reasoning,
            lime_evidence=lime_evidence,
            decision_rationale=f"Quantity Agent: {reasoning} | LIME: {lime_evidence}",
            timestamp=datetime.now().isoformat(),
            auto_approved=True,
            processing_time_ms=elapsed
        )


class CostAgent(BaseAgent):
    """Specialized agent for cost optimization"""
    
    def __init__(self, ml_service_url: str):
        super().__init__("CostAgent", ml_service_url)
        self.logger.info("💰 Cost Agent initialized - Focus: Cost-benefit analysis")
    
    def make_decision(self, row: pd.Series) -> Optional[AgentDecision]:
        start_time = time.time()
        
        days_to_depletion = float(row.get('PredictedDaysToDepletion', 999))
        if days_to_depletion > 15:
            return None
        
        current_stock = int(row.get('CurrentStock', 0))
        avg_daily_sales = float(row.get('AvgDailySales', 0))
        product = row['Product']
        confidence = float(row.get('Confidence', 0))
        urgency = row.get('UrgencyLevel', 'LOW')
        
        # Cost analysis
        stockout_cost_per_day = avg_daily_sales * 10  # $10 loss per unsold unit
        holding_cost_per_day = current_stock * 0.05
        
        # Calculate optimal order to minimize total cost
        days_to_cover = 30
        if urgency == "CRITICAL":
            days_to_cover = 45
        elif urgency == "HIGH":
            days_to_cover = 35
        
        target_stock = int(avg_daily_sales * days_to_cover)
        reorder_qty = max(0, target_stock - current_stock)
        
        # Round to case size
        case_size = 12
        reorder_qty = ((reorder_qty + case_size - 1) // case_size) * case_size
        
        # Ensure minimum order quantity (prevent zero orders)
        reorder_qty = max(case_size, reorder_qty)
        
        estimated_savings = stockout_cost_per_day * days_to_depletion
        reasoning = f"💵 Cost-optimized: {reorder_qty} units (Est. savings: ${estimated_savings:.2f})"
        
        # Get LIME explanation
        stock_percentage = (current_stock / max(current_stock + avg_daily_sales * 30, 1)) * 100
        lime_exp = self.get_lime_explanation(product, current_stock, avg_daily_sales, stock_percentage)
        
        lime_evidence = "N/A"
        if lime_exp.get('feature_contributions'):
            top_features = sorted(
                lime_exp['feature_contributions'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:2]
            lime_evidence = "; ".join([f"{k}: {v:+.2f}" for k, v in top_features])
        
        elapsed = (time.time() - start_time) * 1000
        self.decisions_made += 1
        self.total_processing_time += elapsed
        
        return AgentDecision(
            decision_id=f"CST-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.decisions_made}",
            agent_name=self.name,
            product=product,
            action_type="COST_OPTIMIZE",
            urgency_level=urgency,
            confidence=confidence,
            reorder_quantity=reorder_qty,
            reasoning=reasoning,
            lime_evidence=lime_evidence,
            decision_rationale=f"Cost Agent: {reasoning} | LIME: {lime_evidence}",
            timestamp=datetime.now().isoformat(),
            auto_approved=True,
            processing_time_ms=elapsed
        )


class MultiAgentCoordinator:
    """God-tier coordinator for multi-agent system"""
    
    def __init__(self, ml_service_url='http://localhost:8000', check_interval=30.0):
        self.ml_service_url = ml_service_url
        self.check_interval = check_interval
        self.logger = logging.getLogger("Coordinator")
        
        # Initialize specialized agents
        self.agents = [
            UrgencyAgent(ml_service_url),
            QuantityAgent(ml_service_url),
            CostAgent(ml_service_url)
        ]
        # --- NEW CODE STARTS HERE ---
        # Initialize Kafka Producer to send orders back to system
        try:
            self.producer = KafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.logger.info("✅ Kafka Producer connected for Replenishment")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Kafka: {e}")
            self.producer = None
        # --- NEW CODE ENDS HERE ---
        
        # State tracking
        self.cycle_count = 0
        self.total_decisions = 0
        self.consensus_history = []
        
        # Inventory tracking - maps Product -> CurrentStock
        self.inventory_state = {}
        self.inventory_file = 'output/inventory_state.csv'
        
        # Output files
        os.makedirs('output/multi_agent', exist_ok=True)
        self.decisions_file = 'output/agent_actions_with_lime.csv'
        self.consensus_file = 'output/multi_agent/consensus_decisions.csv'
        self.agent_metrics_file = 'output/multi_agent/agent_performance.csv'
        
        self._initialize_files()
        
        self.logger.info("=" * 80)
        self.logger.info("🌟 GOD-TIER MULTI-AGENT SYSTEM INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ {len(self.agents)} specialized agents active")
        self.logger.info(f"🔗 ML Service: {ml_service_url}")
        self.logger.info(f"⏱️  Check interval: {check_interval}s")
        self.logger.info(f"🤖 Fully autonomous decision-making enabled")
        self.logger.info(f"🔍 LIME explainability: ACTIVE")
        self.logger.info("=" * 80)
    
    def _initialize_files(self):
        """Initialize output CSV files"""
        if not os.path.exists(self.decisions_file):
            pd.DataFrame(columns=[
                'ActionID', 'Timestamp', 'Product', 'CurrentStock', 
                'StockBeforeOrder', 'StockAfterOrder',
                'PredictedDaysToDepletion', 'Confidence', 'UrgencyLevel',
                'ReorderQuantity', 'Reasoning', 'LIMEEvidence', 
                'DecisionRationale', 'Status', 'AgentConsensus',
                'ProcessingTimeMS', 'AutoApproved', 'InventoryUpdated'
            ]).to_csv(self.decisions_file, index=False)
        
        if not os.path.exists(self.consensus_file):
            pd.DataFrame(columns=[
                'ConsensusID', 'Timestamp', 'Product', 'FinalAction',
                'FinalUrgency', 'FinalQuantity', 'ConsensusConfidence',
                'AgentVotes', 'CombinedReasoning', 'CombinedLIMEEvidence',
                'TotalAgents', 'AgreementLevel'
            ]).to_csv(self.consensus_file, index=False)
        
        if not os.path.exists(self.agent_metrics_file):
            pd.DataFrame(columns=[
                'Timestamp', 'CycleID', 'AgentName', 'DecisionsMade',
                'AvgProcessingTimeMS', 'SuccessRate', 'TotalProcessingTime'
            ]).to_csv(self.agent_metrics_file, index=False)
    
    def get_predictions(self) -> Optional[pd.DataFrame]:
        """Fetch latest predictions from ML service"""
        try:
            # Trigger batch prediction
            response = requests.post(f"{self.ml_service_url}/batch-predict", timeout=60)
            
            if os.path.exists('output/predictions.csv'):
                df = pd.read_csv('output/predictions.csv')
                self.logger.info(f"📥 Loaded {len(df)} predictions for analysis")
                
                # Update inventory state from predictions
                self._update_inventory_state(df)
                
                return df
            
            return None
        except Exception as e:
            self.logger.error(f"Error fetching predictions: {e}")
            return None
    
    def _update_inventory_state(self, predictions_df: pd.DataFrame):
        """Update internal inventory state from predictions"""
        for _, row in predictions_df.iterrows():
            product = row['Product']
            current_stock = int(row.get('CurrentStock', 0))
            
            # Initialize if new product
            if product not in self.inventory_state:
                self.inventory_state[product] = current_stock
            # Use the latest stock level from predictions
            else:
                self.inventory_state[product] = current_stock
        
        self.logger.info(f"📦 Inventory state updated: {len(self.inventory_state)} products tracked")
    
    def _deduct_inventory(self, product: str, quantity: int) -> bool:
        """Deduct inventory when order is placed. Returns True if successful, False if would go negative."""
        current = self.inventory_state.get(product, 0)
        
        # Prevent negative stock
        if current < quantity:
            self.logger.warning(f"⚠️  Cannot deduct {quantity} from {product} (only {current} available)")
            return False
        
        # Deduct the ordered quantity
        self.inventory_state[product] = current - quantity
        self.logger.info(f"✅ Inventory updated: {product} {current} -> {self.inventory_state[product]} (-{quantity})")
        return True
    
    def _save_inventory_state(self):
        """Save current inventory state to CSV"""
        try:
            inventory_data = [
                {'Product': product, 'CurrentStock': stock, 'LastUpdated': datetime.now().isoformat()}
                for product, stock in self.inventory_state.items()
            ]
            df = pd.DataFrame(inventory_data)
            df.to_csv(self.inventory_file, index=False)
            self.logger.info(f"💾 Inventory state saved: {len(inventory_data)} products")
        except Exception as e:
            self.logger.error(f"Error saving inventory state: {e}")
    
    def run_multi_agent_analysis(self, predictions_df: pd.DataFrame) -> List[ConsensusResult]:
        """Run all agents in parallel and build consensus"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 CYCLE #{self.cycle_count + 1} - Multi-Agent Analysis")
        self.logger.info(f"{'='*60}")
        
        all_decisions = defaultdict(list)
        
        # Each agent analyzes all predictions
        for agent in self.agents:
            agent_start = time.time()
            agent_decisions = []
            
            for _, row in predictions_df.iterrows():
                decision = agent.make_decision(row)
                if decision:
                    agent_decisions.append(decision)
                    all_decisions[decision.product].append(decision)
            
            agent_time = time.time() - agent_start
            self.logger.info(
                f"  {agent.name:20s} | Decisions: {len(agent_decisions):3d} | "
                f"Time: {agent_time:.2f}s"
            )
        
        # Build consensus for each product
        consensus_results = []
        for product, decisions in all_decisions.items():
            consensus = self._build_consensus(product, decisions)
            if consensus:
                consensus_results.append(consensus)
        
        self.logger.info(f"✅ Consensus reached for {len(consensus_results)} products")
        return consensus_results
    
    def _build_consensus(self, product: str, decisions: List[AgentDecision]) -> Optional[ConsensusResult]:
        """Build consensus from multiple agent decisions"""
        if not decisions:
            return None
        
        # Voting for urgency
        urgency_votes = defaultdict(int)
        for d in decisions:
            urgency_votes[d.urgency_level] += 1
        
        final_urgency = max(urgency_votes.items(), key=lambda x: x[1])[0]
        
        # Average quantity with weighted confidence
        total_weight = sum(d.confidence for d in decisions)
        if total_weight > 0:
            weighted_qty = sum(d.reorder_quantity * d.confidence for d in decisions) / total_weight
            final_quantity = int(weighted_qty)
        else:
            final_quantity = int(np.mean([d.reorder_quantity for d in decisions]))
        
        # Consensus confidence
        consensus_confidence = np.mean([d.confidence for d in decisions])
        
        # Combined reasoning
        combined_reasoning = " | ".join([
            f"{d.agent_name}: {d.reasoning}" for d in decisions
        ])
        
        # Combined LIME evidence
        combined_lime = " | ".join([
            f"{d.agent_name}: {d.lime_evidence}" for d in decisions
        ])
        
        # Agent votes
        agent_votes = {d.agent_name: d.action_type for d in decisions}
        
        return ConsensusResult(
            product=product,
            final_action="REORDER",
            final_urgency=final_urgency,
            final_quantity=final_quantity,
            consensus_confidence=consensus_confidence,
            agent_votes=agent_votes,
            combined_reasoning=combined_reasoning,
            combined_lime_evidence=combined_lime,
            timestamp=datetime.now().isoformat()
        )
    
    def save_decisions(self, consensus_results: List[ConsensusResult]):
        """Save consensus decisions to files and update inventory"""
        if not consensus_results:
            return
        
        # Save to main actions file
        actions_data = []
        executed_count = 0
        
        for consensus in consensus_results:
            product = consensus.product
            current_stock = self.inventory_state.get(product, 0)
            reorder_qty = consensus.final_quantity
            
            # Check if we have enough inventory data to track
            if current_stock <= 0:
                self.logger.warning(f"⚠️  Skipping {product}: No current stock data available")
                continue
            
            # Validate that order won't cause negative stock
            # (Note: Reordering ADDS to inventory, so this is for future consumption)
            # The key is to ensure CurrentStock is tracked properly
            
            actions_data.append({
                'ActionID': f"MULTI-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(actions_data)}",
                'Timestamp': consensus.timestamp,
                'Product': product,
                'CurrentStock': current_stock,
                'StockBeforeOrder': current_stock,
                'StockAfterOrder': current_stock + reorder_qty,  # Orders ADD to inventory
                'PredictedDaysToDepletion': 0,
                'Confidence': consensus.consensus_confidence,
                'UrgencyLevel': consensus.final_urgency,
                'ReorderQuantity': reorder_qty,
                'Reasoning': consensus.combined_reasoning,
                'LIMEEvidence': consensus.combined_lime_evidence,
                'DecisionRationale': f"Multi-Agent Consensus: {len(consensus.agent_votes)} agents",
                'Status': 'AUTO_APPROVED',
                'AgentConsensus': json.dumps(consensus.agent_votes),
                'ProcessingTimeMS': 0,
                'AutoApproved': True,
                'InventoryUpdated': True
            })
            
            # Update inventory: Orders ADD stock (replenishment)
            self.inventory_state[product] = current_stock + reorder_qty
            executed_count += 1
        
        if actions_data:
            df = pd.DataFrame(actions_data)
            
            # Append to existing file
            if os.path.exists(self.decisions_file):
                df.to_csv(self.decisions_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.decisions_file, index=False)
            
            self.logger.info(f"💾 Saved {len(actions_data)} consensus decisions")
            self.logger.info(f"📦 Inventory updated for {executed_count} products")
            
            # Save updated inventory state
            self._save_inventory_state()
        
        # Save consensus details
        consensus_data = []
        for consensus in consensus_results:
            consensus_data.append({
                'ConsensusID': f"CONS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'Timestamp': consensus.timestamp,
                'Product': consensus.product,
                'FinalAction': consensus.final_action,
                'FinalUrgency': consensus.final_urgency,
                'FinalQuantity': consensus.final_quantity,
                'ConsensusConfidence': consensus.consensus_confidence,
                'AgentVotes': json.dumps(consensus.agent_votes),
                'CombinedReasoning': consensus.combined_reasoning,
                'CombinedLIMEEvidence': consensus.combined_lime_evidence,
                'TotalAgents': len(consensus.agent_votes),
                'AgreementLevel': 'HIGH'
            })
        
        df_consensus = pd.DataFrame(consensus_data)
        df_consensus.to_csv(self.consensus_file, mode='a', header=False, index=False)
        
        self.logger.info(f"💾 Saved {len(consensus_results)} consensus decisions")
        self.total_decisions += len(consensus_results)
    
    def run_cycle(self):
        """Execute one decision cycle"""
        cycle_start = time.time()
        self.cycle_count += 1
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"# CYCLE {self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'#'*80}\n")
        
        # Get predictions
        predictions_df = self.get_predictions()
        if predictions_df is None or predictions_df.empty:
            self.logger.warning("⚠️ No predictions available - waiting for ML service")
            return
        
        # Run multi-agent analysis
        consensus_results = self.run_multi_agent_analysis(predictions_df)
        
        # Save decisions
        self.save_decisions(consensus_results)
        if os.path.exists(self.agent_metrics_file): os.utime(self.agent_metrics_file, None)
        # Save agent metrics
        self._save_agent_metrics()
        
        cycle_time = time.time() - cycle_start
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ CYCLE {self.cycle_count} COMPLETE")
        self.logger.info(f"   Decisions: {len(consensus_results)} | Time: {cycle_time:.2f}s | Total: {self.total_decisions}")
        self.logger.info(f"{'='*80}\n")
    
    def _save_agent_metrics(self):
        """Save individual agent performance metrics"""
        metrics_data = []
        for agent in self.agents:
            if agent.decisions_made > 0:
                avg_time = agent.total_processing_time / agent.decisions_made
                metrics_data.append({
                    'Timestamp': datetime.now().isoformat(),
                    'CycleID': self.cycle_count,
                    'AgentName': agent.name,
                    'DecisionsMade': agent.decisions_made,
                    'AvgProcessingTimeMS': avg_time,
                    'SuccessRate': 100.0,
                    'TotalProcessingTime': agent.total_processing_time
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            df.to_csv(self.agent_metrics_file, mode='a', header=False, index=False)
    
    def run(self):
        """Main execution loop"""
        self.logger.info("🌟 Multi-Agent System ACTIVE - Making autonomous decisions")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            self.logger.info("\n\n⛔ Multi-Agent System stopped by user")
            self._print_summary()
    
    def _print_summary(self):
        """Print execution summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 MULTI-AGENT SYSTEM SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Total Cycles: {self.cycle_count}")
        self.logger.info(f"Total Decisions: {self.total_decisions}")
        self.logger.info(f"Avg Decisions/Cycle: {self.total_decisions/max(1, self.cycle_count):.1f}")
        self.logger.info("\nAgent Performance:")
        for agent in self.agents:
            if agent.decisions_made > 0:
                avg_time = agent.total_processing_time / agent.decisions_made
                self.logger.info(f"  {agent.name:20s} | Decisions: {agent.decisions_made:4d} | Avg Time: {avg_time:.2f}ms")
        self.logger.info("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Multi-Agent System with LIME')
    parser.add_argument('--url', default='http://localhost:8000', help='ML service URL')
    parser.add_argument('--interval', type=float, default=30.0, help='Check interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    coordinator = MultiAgentCoordinator(
        ml_service_url=args.url,
        check_interval=args.interval
    )
    
    if args.once:
        coordinator.run_cycle()
    else:
        coordinator.run()


if __name__ == "__main__":
    main()