"""
Kafka Stream Processor for LiveInsight+
Stateful aggregation engine with windowing - replaces Spark DStreams
Implements running totals, time-based windows, and snapshot exports
"""

import os
import json
import time
import logging
import glob
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any
import pandas as pd
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- PRESET INVENTORY CONFIGURATION ---
PRESET_INVENTORY = {
    #"Salt (1kg)": 10000,
    #"Sugar (1kg)": 10000,
    #"Biscuits": 8000,
    #"Kurkure": 5000,
    #"Lays Chips": 5000,
    #"Chocolate": 4000,
    #"Bisleri Water (1L)": 3000,
    #"Pepsi (2L)": 2000,
    #"Coca Cola (2L)": 2000,
    #"Real Juice (1L)": 1500,
    #"Milk (1L)": 500,
    #"Curd (500g)": 400,
    #"Butter (100g)": 400,
    #"Cheese (200g)": 400,
    #"Paneer (200g)": 300,
    #"Rice (5kg)": 1000,
    #"Wheat Flour (5kg)": 1000,
    #"Cooking Oil (1L)": 1200,
    #"Pulses (1kg)": 1500,
    #"Soap (125g)": 2500,
    #"Toothpaste (150g)": 2000,
    #"Facewash (100ml)": 1500,
    #"Shampoo (250ml)": 1500,
    #"Red Bull (250ml)": 800,
    #"Dry Fruits (250g)": 600
}
DEFAULT_STOCK = 1500
# --------------------------------------

class StatefulProcessor:
    """
    Stateful stream processor with windowing and checkpointing
    Demonstrates real-time aggregation without Spark/MapReduce
    """
    
    def __init__(self, bootstrap_servers='localhost:9092', 
                 topic='retail.transactions',
                 checkpoint_interval=5.0):
        """
        Initialize processor with Kafka consumer
        """
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            group_id='liveinsight-processor-v1',
            max_poll_records=500,
            session_timeout_ms=30000
        )
        
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = time.time()
        
        # Initialize Memory State
        self.reset_state()
        
        # Output directory
        os.makedirs('output', exist_ok=True)
        
        logger.info("✅ Stream processor initialized")
        logger.info(f"📡 Consuming from: {topic}")
        logger.info(f"💾 Checkpoint interval: {checkpoint_interval}s")
    
    def reset_state(self):
        """Clear all in-memory aggregation data and reset inventory"""
        logger.warning("🧹 WIPING MEMORY STATE & RESETTING INVENTORY...")
        self.state = {
            'branch_sales': defaultdict(float),
            'category_sales': defaultdict(float),
            'product_sales': defaultdict(int),
            'product_revenue': defaultdict(float),
            'payment_counts': defaultdict(int),
            'branch_time_demand': defaultdict(float),
            'weekly_branch_revenue': defaultdict(float),
            'monthly_branch_revenue': defaultdict(float),
            'product_inventory_usage': defaultdict(int),
            'hourly_transactions': defaultdict(int),
            'customer_transactions': defaultdict(int),
            # Initialize Current Stock with Presets
            'current_inventory': PRESET_INVENTORY.copy()
        }
        
        # Performance metrics reset
        self.metrics = {
            'messages_processed': 0,
            'start_time': time.time(),
            'last_throughput_check': time.time(),
            'messages_since_last_check': 0
        }

    def clear_output_files(self):
        """Delete all CSV files in output directory"""
        logger.warning("🗑️  DELETING OUTPUT FILES...")
        files = glob.glob('output/*.csv')
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                logger.error(f"Could not delete {f}: {e}")
        logger.info("✅ Output directory cleared.")

    def parse_transaction(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and enrich transaction message
        """
        try:
            ts = datetime.strptime(msg['Timestamp'], "%Y-%m-%d %H:%M:%S")
            hour = ts.hour
            week = ts.strftime('%Y-W%U')
            month = ts.strftime('%Y-%m')
            
            # Time slot classification
            if 6 <= hour < 12:
                time_slot = "Morning"
            elif 12 <= hour < 17:
                time_slot = "Afternoon"
            elif 17 <= hour < 21:
                time_slot = "Evening"
            else:
                time_slot = "Night"
            
            return {
                'TransactionID': msg['TransactionID'],
                'Timestamp': ts,
                'BranchName': msg['BranchName'],
                'Category': msg['Category'],
                'Product': msg['Product'],
                'Quantity': msg['Quantity'],
                'FinalAmount': msg['FinalAmount'],
                'PaymentType': msg['PaymentType'],
                'CustomerID': msg['CustomerID'],
                'TimeSlot': time_slot,
                'Week': week,
                'Month': month,
                'Hour': hour
            }
        except Exception as e:
            logger.error(f"Parse error: {e} | Message: {msg}")
            return None
    
    def update_state(self, txn: Dict[str, Any]):
        """
        Update all aggregation state stores
        """
        product = txn['Product']
        qty = txn['Quantity']

        # Branch sales
        self.state['branch_sales'][txn['BranchName']] += txn['FinalAmount']
        
        # Category sales
        self.state['category_sales'][txn['Category']] += txn['FinalAmount']
        
        # Product sales (units)
        self.state['product_sales'][product] += qty
        
        # Product revenue
        self.state['product_revenue'][product] += txn['FinalAmount']
        
        # Payment type counts
        self.state['payment_counts'][txn['PaymentType']] += 1
        
        # Branch-TimeSlot demand
        key = (txn['BranchName'], txn['TimeSlot'])
        self.state['branch_time_demand'][key] += txn['FinalAmount']
        
        # Weekly branch revenue
        key = (txn['BranchName'], txn['Week'])
        self.state['weekly_branch_revenue'][key] += txn['FinalAmount']
        
        # Monthly branch revenue
        key = (txn['BranchName'], txn['Month'])
        self.state['monthly_branch_revenue'][key] += txn['FinalAmount']
        
        # Product inventory usage (total units sold)
        self.state['product_inventory_usage'][product] += qty
        
        # --- LIVE INVENTORY UPDATE ---
        # Initialize if product not in preset
        if product not in self.state['current_inventory']:
            self.state['current_inventory'][product] = DEFAULT_STOCK
        
        # Decrement stock
        self.state['current_inventory'][product] -= qty
        # -----------------------------

        # Hourly transaction counts
        hour_key = txn['Timestamp'].strftime('%Y-%m-%d %H:00')
        self.state['hourly_transactions'][hour_key] += 1
        
        # Customer transaction counts
        self.state['customer_transactions'][txn['CustomerID']] += 1
    
    def checkpoint_state(self):
        """Write state snapshots to CSV files"""
        try:
            # Branch sales
            df = pd.DataFrame([
                {'BranchName': k, 'TotalRevenue': v}
                for k, v in self.state['branch_sales'].items()
            ])
            df.to_csv('output/branch_sales.csv', index=False)
            
            # Category sales
            df = pd.DataFrame([
                {'Category': k, 'TotalRevenue': v}
                for k, v in self.state['category_sales'].items()
            ])
            df.to_csv('output/category_sales.csv', index=False)
            
            # Product sales (units)
            df = pd.DataFrame([
                {'Product': k, 'UnitsSold': v}
                for k, v in self.state['product_sales'].items()
            ])
            df.to_csv('output/product_sales.csv', index=False)
            
            # Payment type analysis
            df = pd.DataFrame([
                {'PaymentType': k, 'TransactionCount': v}
                for k, v in self.state['payment_counts'].items()
            ])
            df.to_csv('output/payment_type_analysis.csv', index=False)
            
            # Branch-time demand
            df = pd.DataFrame([
                {'BranchName': k[0], 'TimeSlot': k[1], 'TotalRevenue': v}
                for k, v in self.state['branch_time_demand'].items()
            ])
            df.to_csv('output/branch_time_demand.csv', index=False)
            
            # Weekly branch revenue
            df = pd.DataFrame([
                {'BranchName': k[0], 'Week': k[1], 'TotalRevenue': v}
                for k, v in self.state['weekly_branch_revenue'].items()
            ])
            df.to_csv('output/weekly_branch_revenue.csv', index=False)
            
            # Monthly branch revenue
            df = pd.DataFrame([
                {'BranchName': k[0], 'Month': k[1], 'TotalRevenue': v}
                for k, v in self.state['monthly_branch_revenue'].items()
            ])
            df.to_csv('output/monthly_branch_revenue.csv', index=False)
            
            # Product inventory usage (for ML service)
            # Also including Current Stock here for the ML model!
            inventory_data = []
            for product, sold in self.state['product_inventory_usage'].items():
                current = self.state['current_inventory'].get(product, DEFAULT_STOCK)
                inventory_data.append({
                    'Product': product,
                    'TotalUnitsSold': sold,
                    'CurrentStock': current
                })
            
            df = pd.DataFrame(inventory_data)
            df.to_csv('output/product_inventory_usage.csv', index=False)
            
            # Hourly transactions
            df = pd.DataFrame([
                {'Hour': k, 'TransactionCount': v}
                for k, v in self.state['hourly_transactions'].items()
            ])
            df.to_csv('output/hourly_transactions.csv', index=False)
            
            logger.info("💾 Checkpoint saved (9 aggregates)")
            
        except Exception as e:
            logger.error(f"Checkpoint error: {e}", exc_info=True)
    
    def log_metrics(self):
        """Log throughput and latency metrics"""
        now = time.time()
        elapsed = now - self.metrics['start_time']
        total_msgs = self.metrics['messages_processed']
        
        # Calculate throughput since last check
        time_since_check = now - self.metrics['last_throughput_check']
        if time_since_check >= 10.0:  # Log every 10s
            recent_msgs = self.metrics['messages_since_last_check']
            throughput = recent_msgs / time_since_check
            
            logger.info("=" * 70)
            logger.info(f"📊 METRICS | Total: {total_msgs} msgs | "
                       f"Uptime: {elapsed:.1f}s | "
                       f"Throughput: {throughput:.2f} msg/s")
            logger.info("=" * 70)
            
            self.metrics['last_throughput_check'] = now
            self.metrics['messages_since_last_check'] = 0
    
    def run(self):
        """Main processing loop"""
        logger.info("🚀 Stream processor started")
        logger.info("⏳ Waiting for messages...")
        
        try:
            for message in self.consumer:
                msg_data = message.value

                # 1. CHECK FOR RESET COMMAND
                if msg_data.get('control_message') == True and msg_data.get('type') == 'RESET':
                    logger.warning("🚨 RECEIVED RESET COMMAND FROM PRODUCER")
                    self.reset_state()
                    self.clear_output_files()
                    self.checkpoint_state()
                    continue 

                # 2. CHECK FOR REPLENISHMENT (FROM AGENT) - [NEW LOGIC HERE]
                if msg_data.get('type') == 'REPLENISHMENT':
                    product = msg_data.get('product')
                    qty = msg_data.get('quantity')
                    
                    # Increase Stock
                    if product in self.state['current_inventory']:
                        self.state['current_inventory'][product] += qty
                    else:
                        self.state['current_inventory'][product] = qty
                        
                    logger.info(f"🚚 RESTOCK RECEIVED: {product} (+{qty} units)")
                    
                    # Force a checkpoint so Dashboard sees it immediately
                    self.checkpoint_state()
                    continue # Skip sales parsing for this message

                # 3. PROCESS SALES TRANSACTIONS
                txn = self.parse_transaction(msg_data)
                if txn is None:
                    continue
                
                # Update state (Sales logic)
                self.update_state(txn)
                
                # Update metrics
                self.metrics['messages_processed'] += 1
                self.metrics['messages_since_last_check'] += 1
                
                # Checkpoint if needed
                now = time.time()
                if now - self.last_checkpoint >= self.checkpoint_interval:
                    self.checkpoint_state()
                    self.last_checkpoint = now
                
                self.log_metrics()
                
        except KeyboardInterrupt:
            logger.info("⚠️  Received shutdown signal")
        except Exception as e:
            logger.error(f"❌ Processing error: {e}", exc_info=True)
        finally:
            self.checkpoint_state()
            self.consumer.close()
            logger.info("✅ Processor shutdown complete")
            
            # Final metrics
            elapsed = time.time() - self.metrics['start_time']
            total = self.metrics['messages_processed']
            avg_throughput = total / elapsed if elapsed > 0 else 0
            
            logger.info("=" * 70)
            logger.info("📈 FINAL STATS")
            logger.info(f"   Total messages: {total}")
            logger.info(f"   Total time: {elapsed:.2f}s")
            logger.info(f"   Avg throughput: {avg_throughput:.2f} msg/s")
            logger.info("=" * 70)
            logger.info("✅ Processor shutdown complete")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka Stream Processor')
    parser.add_argument('--broker', default='localhost:9092', help='Kafka broker')
    parser.add_argument('--topic', default='retail.transactions', help='Topic to consume')
    parser.add_argument('--checkpoint', type=float, default=3.0, help='Checkpoint interval (seconds)')
    
    args = parser.parse_args()
    
    processor = StatefulProcessor(
        bootstrap_servers=args.broker,
        topic=args.topic,
        checkpoint_interval=args.checkpoint
    )
    
    processor.run()


if __name__ == "__main__":
    main()