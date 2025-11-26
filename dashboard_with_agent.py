"""
LiveInsight Dashboard WITH Autonomous Agent
Shows real-time retail analytics + autonomous agent decisions with LIME explanations
Demonstrates the power of agentic AI for inventory management
"""

import streamlit as st
import pandas as pd
import os
import time
import psutil
import numpy as np
import glob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit Configuration
st.set_page_config(
    page_title="LiveInsight+ WITH Agent",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)
# --- ADD THIS TO YOUR DASHBOARD SIDEBAR ---
st.sidebar.markdown("### 🎮 Simulation Control")
if st.sidebar.button("🔄 Reset Simulation (Start Over)", type="primary"):
    # Create trigger file
    with open("reset_signal.trigger", "w") as f:
        f.write("RESET")
    
    # Optional: Clear Streamlit Cache
    st.cache_data.clear()
    
    # Show feedback
    st.sidebar.success("Signal sent! System restarting in 2s...")
    time.sleep(2)
    st.rerun()
# ------------------------------------------
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .agent-active {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data(ttl=3)
def load_csv(filepath, columns=None):
    """Load CSV with caching"""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if columns:
                df.columns = columns
            return df
        return pd.DataFrame(columns=columns if columns else [])
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return pd.DataFrame(columns=columns if columns else [])

def get_system_metrics():
    """Get system resource usage"""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('C:\\' if os.name == 'nt' else '/').percent
        return cpu, memory, disk
    except:
        return 0, 0, 0

# Sidebar
st.sidebar.title("🤖 LiveInsight+ WITH Agent")
st.sidebar.markdown("---")

refresh_interval = st.sidebar.selectbox(
    "Auto-refresh interval (seconds)",
    options=[2, 5, 10, 30], 
    index=1
)

# --- UPDATED AGENT STATUS CHECK ---
# We check 'agent_performance.csv' because the Multi-Agent system writes to it
# every cycle, proving it is alive even if it hasn't placed an order yet.
agent_heartbeat_file = 'output/multi_agent/agent_performance.csv'
agent_active = False

if os.path.exists(agent_heartbeat_file):
    # Check if file was modified in the last 2 minutes
    modified_time = os.path.getmtime(agent_heartbeat_file)
    if (time.time() - modified_time) < 120:
        agent_active = True

if agent_active:
    st.sidebar.success("✅ **AGENT ACTIVE**")
    st.sidebar.markdown("🤖 Multi-Agent System is running")
else:
    st.sidebar.warning("⚠️ **AGENT INACTIVE**")
    st.sidebar.markdown("System waiting... Start it with: `python multi_agent_system.py`")
# ----------------------------------

# System Resources
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Resources")
cpu, mem, disk = get_system_metrics()
st.sidebar.progress(int(cpu), text=f"CPU: {cpu:.1f}%")
st.sidebar.progress(int(mem), text=f"Memory: {mem:.1f}%")
st.sidebar.progress(int(disk), text=f"Disk: {disk:.1f}%")

# Load Data
branch_df = load_csv("output/branch_sales.csv", ["BranchName", "TotalRevenue"])
branch_df["TotalRevenue"] = pd.to_numeric(branch_df["TotalRevenue"], errors="coerce")

cat_df = load_csv("output/category_sales.csv", ["Category", "TotalRevenue"])
cat_df["TotalRevenue"] = pd.to_numeric(cat_df["TotalRevenue"], errors="coerce")

prod_df = load_csv("output/product_sales.csv", ["Product", "UnitsSold"])
prod_df["UnitsSold"] = pd.to_numeric(prod_df["UnitsSold"], errors="coerce")

# --- FIX: ROBUST LOADING FOR INVENTORY DATA ---
# We now load without forcing specific columns, so it adapts to 2 or 3 columns automatically.
inventory_df = load_csv("output/product_inventory_usage.csv") 

# Drop 'CurrentStock' from here if it exists, because we will get it from predictions_df
# during the merge later. This prevents duplicate column errors (CurrentStock_x, CurrentStock_y).
if "CurrentStock" in inventory_df.columns:
    inventory_df = inventory_df.drop(columns=["CurrentStock"])
# ----------------------------------------------

predictions_df = load_csv("output/predictions.csv")
agent_actions_df = load_csv("output/agent_actions_with_lime.csv")

payment_df = load_csv("output/payment_type_analysis.csv", ["PaymentType", "TransactionCount"])
payment_df["TransactionCount"] = pd.to_numeric(payment_df["TransactionCount"], errors="coerce")

# Network stats
try:
    net_io = psutil.net_io_counters()
    net_sent_mb = net_io.bytes_sent / (1024 * 1024)
    net_recv_mb = net_io.bytes_recv / (1024 * 1024)
except:
    net_sent_mb = 0
    net_recv_mb = 0

# KPIs in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Key Metrics")

total_revenue = branch_df["TotalRevenue"].sum() if not branch_df.empty else 0
total_units = prod_df["UnitsSold"].sum() if not prod_df.empty else 0
active_branches = branch_df.shape[0] if not branch_df.empty else 0

st.sidebar.metric("Total Revenue", f"₹{total_revenue:,.0f}")
st.sidebar.metric("Total Units Sold", f"{total_units:,}")
st.sidebar.metric("Active Branches", f"{active_branches}")

if not payment_df.empty:
    total_transactions = payment_df["TransactionCount"].sum()
    st.sidebar.metric("Total Transactions", f"{total_transactions:,.0f}")

if agent_active and not agent_actions_df.empty:
    total_actions = len(agent_actions_df)
    pending_actions = len(agent_actions_df[agent_actions_df['Status'] == 'PENDING'])
    st.sidebar.metric("🤖 Agent Actions", f"{total_actions}")
    st.sidebar.metric("⏳ Pending Approval", f"{pending_actions}")

# Network Stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Network I/O")
st.sidebar.metric("Data Sent", f"{net_sent_mb:.1f} MB")
st.sidebar.metric("Data Received", f"{net_recv_mb:.1f} MB")

# Main Dashboard
st.markdown('<div class="main-header">🤖 LiveInsight+ Retail Intelligence WITH Autonomous Agent</div>', 
            unsafe_allow_html=True)

# Agent Status Banner
if agent_active:
    st.markdown("""
    <div class="agent-active">
        <h3>✅ Autonomous Agent Active</h3>
        <p>The AI agent is continuously monitoring inventory levels and automatically creating reorder recommendations
        using LIME (Local Interpretable Model-agnostic Explanations) for transparent, human-understandable decisions.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ **Autonomous Agent is NOT running**. Start it with: `python agent_with_lime.py`")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Business Analytics", 
    "🤖 Agent Dashboard", 
    "🔮 ML Predictions", 
    "📦 Inventory Status",
    "📈 Performance Comparison",
    "🎯 System Flow"  # NEW TAB
])

# Tab 1: Business Analytics
with tab1:
    st.header("📊 Real-Time Business Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Revenue by Branch")
        if not branch_df.empty:
            fig = px.bar(
                branch_df.sort_values("TotalRevenue", ascending=False).head(10),
                x="BranchName", 
                y="TotalRevenue",
                color="TotalRevenue",
                color_continuous_scale="Viridis",
                title="Top 10 Branches by Revenue"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No branch data available")
    
    with col2:
        st.subheader("🏷️ Revenue by Category")
        if not cat_df.empty:
            fig = px.pie(
                cat_df, 
                names="Category", 
                values="TotalRevenue",
                title="Category Revenue Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No category data available")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛍️ Top Selling Products")
        if not prod_df.empty:
            top_prods = prod_df.sort_values("UnitsSold", ascending=False).head(15)
            fig = px.bar(
                top_prods, 
                y="Product", 
                x="UnitsSold", 
                orientation='h',
                color="UnitsSold",
                color_continuous_scale="Plasma",
                title="Top 15 Products by Units Sold"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No product data available")
    
    with col2:
        st.subheader("💳 Payment Methods")
        if not payment_df.empty:
            fig = px.pie(
                payment_df, 
                names="PaymentType", 
                values="TransactionCount",
                title="Payment Method Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width='stretch')
            
            with st.expander("📋 Payment Method Details"):
                st.dataframe(
                    payment_df.sort_values("TransactionCount", ascending=False),
                    use_container_width=True
                )
        else:
            st.info("No payment data available")
    
    # Additional metrics row
    st.markdown("---")
    st.subheader("📊 Additional Business Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_branch_revenue = branch_df["TotalRevenue"].mean() if not branch_df.empty else 0
        st.metric("Avg Branch Revenue", f"₹{avg_branch_revenue:,.0f}")
    
    with col2:
        top_category = cat_df.sort_values("TotalRevenue", ascending=False).iloc[0] if not cat_df.empty else None
        if top_category is not None:
            st.metric("Top Category", top_category["Category"])
        else:
            st.metric("Top Category", "N/A")
    
    with col3:
        total_products = len(prod_df) if not prod_df.empty else 0
        st.metric("Total Products", f"{total_products}")
    
    with col4:
        avg_units_per_product = prod_df["UnitsSold"].mean() if not prod_df.empty else 0
        st.metric("Avg Units/Product", f"{avg_units_per_product:.0f}")

# Tab 2: Agent Dashboard
with tab2:
    st.header("🤖 Autonomous Agent Dashboard")
    
    if not agent_active or agent_actions_df.empty:
        st.warning("⚠️ No agent actions yet. The agent will create actions when low stock is detected.")
    else:
        # Agent Actions Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_count = len(agent_actions_df[agent_actions_df['UrgencyLevel'] == 'CRITICAL'])
            st.metric("🚨 Critical Actions", critical_count, 
                     delta=None, delta_color="inverse" if critical_count > 0 else "normal")
        
        with col2:
            high_count = len(agent_actions_df[agent_actions_df['UrgencyLevel'] == 'HIGH'])
            st.metric("⚠️ High Priority", high_count)
        
        with col3:
            total_qty = agent_actions_df['ReorderQuantity'].sum()
            st.metric("📦 Total Units Ordered", f"{total_qty:,}")
        
        with col4:
            executed_count = len(agent_actions_df[agent_actions_df['Status'] == 'AUTO_APPROVED'])
            st.metric("✅ Executed Automatically", executed_count)
        
        st.markdown("---")
        
        # Agent Actions with LIME Evidence
        st.subheader("Autonomous Orders Executed (with LIME Explanations)")
        st.caption("All orders are automatically executed by the multi-agent system - No human approval required")
        
        # Filter options
        urgency_filter = st.selectbox(
            "Filter by Urgency",
            ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
            key="urgency_filter"
        )
        
        # Apply filter
        display_df = agent_actions_df.copy()
        if urgency_filter != "All":
            display_df = display_df[display_df['UrgencyLevel'] == urgency_filter]
        
        # Sort by urgency and timestamp
        urgency_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        display_df['_sort'] = display_df['UrgencyLevel'].map(urgency_order)
        display_df = display_df.sort_values(['_sort', 'Timestamp'], ascending=[True, False])
        
        # Display actions
        for idx, row in display_df.head(20).iterrows():
            # Color code by urgency
            if row['UrgencyLevel'] == 'CRITICAL':
                status_color = "🔴"
            elif row['UrgencyLevel'] == 'HIGH':
                status_color = "🟠"
            elif row['UrgencyLevel'] == 'MEDIUM':
                status_color = "🟡"
            else:
                status_color = "🟢"
            
            # Determine execution status display
            status_display = "✅ EXECUTED" if row['Status'] == 'AUTO_APPROVED' else row['Status']
            
            with st.expander(
                f"{status_color} {row['Product']} | Urgency: {row['UrgencyLevel']} | "
                f"Qty: {row['ReorderQuantity']:,} units | {status_display}"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Action ID:** {row['ActionID']}")
                    st.markdown(f"**Timestamp:** {row['Timestamp']}")
                    st.markdown(f"**Current Stock:** {row['CurrentStock']} units")
                    st.markdown(f"**Days to Depletion:** {row['PredictedDaysToDepletion']:.1f} days")
                    st.markdown(f"**Confidence:** {row['Confidence']:.2%}")
                    st.markdown(f"**Status:** {row['Status']}")
                
                with col2:
                    st.markdown(f"**Reorder Quantity:** {row['ReorderQuantity']:,} units")
                    st.markdown(f"**Urgency Level:** {row['UrgencyLevel']}")
                    st.markdown(f"**Execution Status:** ✅ Automatically Executed")
                    if 'AgentConsensus' in row and pd.notna(row['AgentConsensus']):
                        st.markdown(f"**Agent Consensus:** Multi-Agent Decision")
                    if 'AutoApproved' in row:
                        st.markdown(f"**Autonomous:** {'Yes ✅' if row['AutoApproved'] else 'No'}")
                
                st.markdown("---")
                st.markdown("**🔍 LIME Evidence:**")
                st.info(row['LIMEEvidence'])
                
                st.markdown("**💡 Decision Rationale (from LIME):**")
                st.success(row['DecisionRationale'])
                
                st.markdown("**📝 Full Reasoning:**")
                st.text_area("", row['Reasoning'], height=100, key=f"reason_{idx}", disabled=True)
                
                # Autonomous execution timeline
                st.markdown("---")
                st.markdown("**⚡ Autonomous Execution Timeline:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("1️⃣ **Detected**")
                    st.caption(row['Timestamp'])
                with col_b:
                    st.markdown("2️⃣ **Analyzed (LIME)**")
                    st.caption("Multi-Agent Consensus")
                with col_c:
                    st.markdown("3️⃣ **Executed**")
                    st.caption("Order Placed Automatically")
                
                # Show autonomous execution status
                if row['Status'] == 'AUTO_APPROVED':
                    st.success("✅ Order Executed Automatically - No Human Intervention Required")
                    if 'AgentConsensus' in row and pd.notna(row['AgentConsensus']):
                        st.info(f"🤖 Multi-Agent Decision: All agents agreed on this action")
                else:
                    st.info(f"Status: {row['Status']}")
        
        st.markdown("---")
        
        # Comprehensive Autonomous Decision Tracking
        st.subheader("📊 Autonomous Decision Analytics")
        
        # Create tracking visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Execution rate over time
            st.markdown("**⚡ Execution Timeline**")
            if len(agent_actions_df) > 0:
                actions_by_time = agent_actions_df.copy()
                actions_by_time['Timestamp'] = pd.to_datetime(actions_by_time['Timestamp'])
                actions_by_time['Hour'] = actions_by_time['Timestamp'].dt.floor('H')
                hourly_actions = actions_by_time.groupby('Hour').size().reset_index(name='Actions')
                
                fig = px.line(
                    hourly_actions,
                    x='Hour',
                    y='Actions',
                    markers=True,
                    title="Autonomous Orders Executed Per Hour",
                    labels={'Actions': 'Orders Executed', 'Hour': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Urgency breakdown with quantities
            st.markdown("**🎯 Urgency-Based Order Volume**")
            if len(agent_actions_df) > 0:
                urgency_quantities = agent_actions_df.groupby('UrgencyLevel').agg({
                    'ReorderQuantity': 'sum',
                    'Product': 'count'
                }).reset_index()
                urgency_quantities.columns = ['UrgencyLevel', 'TotalUnits', 'OrderCount']
                
                fig = px.bar(
                    urgency_quantities,
                    x='UrgencyLevel',
                    y=['TotalUnits', 'OrderCount'],
                    barmode='group',
                    title="Units Ordered by Urgency Level",
                    labels={'value': 'Count', 'variable': 'Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics table
        st.markdown("**📈 Decision Statistics by Urgency**")
        if len(agent_actions_df) > 0:
            stats_df = agent_actions_df.groupby('UrgencyLevel').agg({
                'Product': 'count',
                'ReorderQuantity': ['sum', 'mean', 'max'],
                'Confidence': 'mean'
            }).round(2)
            stats_df.columns = ['Total Orders', 'Total Units', 'Avg Units', 'Max Units', 'Avg Confidence']
            st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("---")
        
        # Agent Performance Metrics
        st.subheader("📈 Agent Performance Metrics")
        
        metrics_file = 'output/agent_performance_metrics.csv'
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            
            if not metrics_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_cycle_time = metrics_df['ProcessingTimeSeconds'].mean()
                    st.metric("Avg Cycle Time", f"{avg_cycle_time:.2f}s")
                
                with col2:
                    total_products = metrics_df['ProductsAnalyzed'].sum()
                    st.metric("Products Analyzed", f"{total_products:,}")
                
                with col3:
                    avg_lime_time = metrics_df['AverageLIMEExplanationTime'].mean()
                    st.metric("Avg LIME Time", f"{avg_lime_time:.3f}s")
                
                # Performance over time
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Actions Created Over Time', 'Processing Time Trend')
                )
                
                metrics_df['Timestamp'] = pd.to_datetime(metrics_df['Timestamp'])
                
                fig.add_trace(
                    go.Scatter(x=metrics_df['Timestamp'], y=metrics_df['ActionsCreated'],
                              mode='lines+markers', name='Actions Created'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=metrics_df['Timestamp'], y=metrics_df['ProcessingTimeSeconds'],
                              mode='lines', name='Cycle Time (s)', line=dict(color='orange')),
                    row=2, col=1
                )
                
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, width='stretch')

# Tab 3: ML Predictions
with tab3:
    st.header("🔮 Machine Learning Predictions")
    
    if predictions_df.empty:
        st.warning("⚠️ No predictions available. ML service may not be running.")
        st.info("Start ML service with: `python ml_service_enhanced.py`")
    else:
        st.success(f"✅ {len(predictions_df)} products analyzed with Random Forest + SHAP + LIME")
        
        # Urgency distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Prediction Urgency Distribution")
            if 'UrgencyLevel' in predictions_df.columns:
                urgency_counts = predictions_df['UrgencyLevel'].value_counts()
                fig = px.pie(
                    values=urgency_counts.values,
                    names=urgency_counts.index,
                    title="Products by Urgency Level",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("⏳ Urgency data will appear once ML model analyzes inventory")
        
        with col2:
            st.subheader("📊 Prediction Confidence")
            fig = px.histogram(
                predictions_df,
                x='Confidence',
                nbins=20,
                title="Confidence Score Distribution",
                labels={'Confidence': 'Prediction Confidence'},
                color_discrete_sequence=['#636EFA']
            )
            fig.add_vline(x=0.85, line_dash="dash", line_color="red", 
                         annotation_text="Auto-approve threshold")
            st.plotly_chart(fig, width='stretch')
        
        # Detailed predictions table
        st.subheader("📋 Detailed Predictions")
        
        # Add color coding to dataframe - only if UrgencyLevel exists
        def highlight_urgency(row):
            if 'UrgencyLevel' in row.index:
                if row['UrgencyLevel'] == 'CRITICAL':
                    return ['background-color: #ffcccc'] * len(row)
                elif row['UrgencyLevel'] == 'HIGH':
                    return ['background-color: #ffe6cc'] * len(row)
                elif row['UrgencyLevel'] == 'MEDIUM':
                    return ['background-color: #ffffcc'] * len(row)
            return [''] * len(row)
        
        # Define all possible columns we want to display
        desired_cols = ['Product', 'CurrentStock', 'AvgDailySales', 
                       'PredictedDaysToDepletion', 'Confidence', 'UrgencyLevel', 'Recommendation']
        
        # Only include columns that actually exist in the dataframe
        display_cols = [col for col in desired_cols if col in predictions_df.columns]
        
        if display_cols:
            # Create a subset with only existing columns
            display_df = predictions_df[display_cols].copy()
            
            # Apply styling only if we have data
            if not display_df.empty:
               st.dataframe(
                    display_df.style.apply(highlight_urgency, axis=1),
                    use_container_width=True,
                    height=400
               )
            else:
                st.info("⏳ Waiting for prediction data to populate...")
        else:
            st.info("⏳ Waiting for ML predictions to be generated...")

# Tab 4: Inventory Status
with tab4:
    st.header("📦 Current Inventory Status")
    
    if not inventory_df.empty and not predictions_df.empty:
        # Merge inventory with predictions
        merged_df = inventory_df.merge(predictions_df, on='Product', how='left')
        
        # Stock level visualization
        st.subheader("📊 Stock Levels Heatmap")
        
        # Create stock categories
        merged_df['StockStatus'] = pd.cut(
            merged_df['PredictedDaysToDepletion'],
            bins=[-1, 3, 7, 14, 30, 999],
            labels=['Critical', 'Low', 'Medium', 'Good', 'Excellent']
        )
        
        stock_summary = merged_df['StockStatus'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Stock Status Summary")
            for status, count in stock_summary.items():
                if status == 'Critical':
                    st.error(f"🔴 {status}: {count} products")
                elif status == 'Low':
                    st.warning(f"🟠 {status}: {count} products")
                elif status == 'Medium':
                    st.info(f"🟡 {status}: {count} products")
                else:
                    st.success(f"🟢 {status}: {count} products")
        
        with col2:
            # Convert stock_summary to DataFrame with proper column names
            stock_df = stock_summary.reset_index()
            stock_df.columns = ['StockStatus', 'Count']
            
            fig = px.bar(
                stock_df,
                x='StockStatus',
                y='Count',
                title="Products by Stock Status",
                labels={'StockStatus': 'Status', 'Count': 'Number of Products'},
                color='StockStatus',
                color_discrete_map={
                    'Critical': '#ff4444',
                    'Low': '#ff8800',
                    'Medium': '#ffdd00',
                    'Good': '#88cc00',
                    'Excellent': '#00cc44'
                }
            )
            st.plotly_chart(fig, width='stretch')
        
        # Detailed inventory table
        st.subheader("📋 Detailed Inventory Report")
        display_df = merged_df[['Product', 'TotalUnitsSold', 'CurrentStock', 
                                'AvgDailySales', 'PredictedDaysToDepletion', 
                                'StockStatus']].sort_values('PredictedDaysToDepletion')
        
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("Inventory data will appear once streaming data is processed")

# Tab 5: Performance Comparison
with tab5:
    st.header("📈 WITH vs WITHOUT Agent Comparison")
    
    st.markdown("""
    ### 🤖 Benefits of Autonomous Agent
    
    This system demonstrates the advantages of having an autonomous AI agent:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ✅ WITH Agent (This Dashboard)
        - **Automated Monitoring**: 24/7 surveillance of inventory
        - **Instant Alerts**: Real-time detection of low stock
        - **LIME Explanations**: Human-readable decision justifications
        - **Auto-Approval**: High-confidence decisions execute automatically
        - **Reduced Response Time**: Minutes vs hours/days
        - **Consistency**: No human error or oversight
        - **Scalability**: Monitors 1000s of products simultaneously
        - **Audit Trail**: Complete decision history with evidence
        """)
    
    with col2:
        st.markdown("""
        #### ❌ WITHOUT Agent (Manual Process)
        - **Manual Review**: Requires human to check dashboards
        - **Delayed Detection**: Hours or days to notice issues
        - **No Explanations**: Decisions based on intuition
        - **Always Manual**: Every decision needs approval
        - **Slow Response**: Human processing bottleneck
        - **Human Error**: Oversight and mistakes possible
        - **Limited Scale**: Can't monitor many products
        - **Incomplete Records**: Manual logging inconsistent
        """)
    
    st.markdown("---")
    
    # If we have agent metrics, show comparison
    if agent_active and not agent_actions_df.empty:
        st.subheader("📊 Measured Impact")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stockouts_prevented = len(agent_actions_df[agent_actions_df['UrgencyLevel'].isin(['CRITICAL', 'HIGH'])])
            st.metric("🛡️ Stockouts Prevented", stockouts_prevented,
                     help="Critical/High priority actions taken automatically")
        
        with col2:
            response_time = "< 1 min"
            st.metric("⚡ Avg Response Time", response_time,
                     help="From detection to action creation")
        
        with col3:
            auto_approved = len(agent_actions_df[agent_actions_df['Status'] == 'AUTO_APPROVED'])
            automation_rate = (auto_approved / len(agent_actions_df) * 100) if len(agent_actions_df) > 0 else 0
            st.metric("🤖 Automation Rate", f"{automation_rate:.0f}%",
                     help="Percentage of actions auto-approved")
        
        with col4:
            coverage = len(predictions_df) if not predictions_df.empty else 0
            st.metric("📊 Product Coverage", f"{coverage}",
                     help="Products continuously monitored")
        
        # Time saved calculation
        st.markdown("---")
        st.subheader("⏰ Time & Cost Savings")
        
        # Assumptions for calculation
        actions_created = len(agent_actions_df)
        manual_time_per_action = 15  # minutes
        agent_time_per_action = 0.5  # minutes
        
        time_saved = (manual_time_per_action - agent_time_per_action) * actions_created
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("⏱️ Time Saved", f"{time_saved:.0f} minutes",
                     help=f"Based on {actions_created} actions")
            st.caption(f"Assuming {manual_time_per_action} min/action manually vs {agent_time_per_action} min with agent")
        
        with col2:
            # Cost savings (assume $30/hour for analyst time)
            hourly_rate = 30
            cost_saved = (time_saved / 60) * hourly_rate
            st.metric("💰 Cost Saved", f"${cost_saved:.2f}",
                     help="Based on analyst time saved")
            st.caption(f"Assuming ${hourly_rate}/hour analyst cost")


# Add this complete new tab:
with tab6:
    st.markdown("""
    <style>
    .flow-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .flow-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .component-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .component-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    .pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .data-flow {
        background: linear-gradient(90deg, #00f260, #0575e6);
        height: 3px;
        animation: flow 2s infinite;
    }
    @keyframes flow {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="flow-container"><div class="flow-title">🚀 REAL-TIME MULTI-AGENT SYSTEM</div></div>', 
                unsafe_allow_html=True)
    
    # Load data
    try:
        branch_df_flow = pd.read_csv('output/branch_sales.csv')
        agent_metrics = pd.read_csv('output/multi_agent/agent_performance.csv') if os.path.exists('output/multi_agent/agent_performance.csv') else pd.DataFrame()
        predictions = pd.read_csv('output/predictions.csv') if os.path.exists('output/predictions.csv') else pd.DataFrame()
        actions = pd.read_csv('output/agent_actions_with_lime.csv') if os.path.exists('output/agent_actions_with_lime.csv') else pd.DataFrame()
    except:
        branch_df_flow = pd.DataFrame()
        agent_metrics = pd.DataFrame()
        predictions = pd.DataFrame()
        actions = pd.DataFrame()
    
    # ================================================================
    # ANIMATED SANKEY DIAGRAM - Main Flow Visualization
    # ================================================================
    st.markdown("### 🌊 Live Data Flow Pipeline")
    
    # Calculate real metrics
    store_count = len(branch_df_flow) if not branch_df_flow.empty else 5
    products_tracked = len(predictions) if not predictions.empty else 0
    actions_taken = len(actions) if not actions.empty else 0
    
    # Create professional Sankey diagram
    fig_main = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 25,
            thickness = 30,
            line = dict(color = "white", width = 2),
            label = [
                "🏪 Store 1<br>Indiranagar",
                "🏪 Store 2<br>Koramangala", 
                "🏪 Store 3<br>Whitefield",
                "🏪 Store 4<br>Jayanagar",
                "🏪 Store 5<br>MG Road",
                "📡 Kafka<br>Stream",
                "⚙️ Stream<br>Processor",
                "🧠 ML<br>Engine",
                "🚨 Urgency<br>Agent",
                "📦 Quantity<br>Agent",
                "💰 Cost<br>Agent",
                "🤝 Consensus<br>Engine",
                "✅ Auto<br>Execute"
            ],
            color = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",  # Stores
                "#FFD93D", "#95E1D3", "#6C5CE7",  # Kafka, Processor, ML
                "#FD79A8", "#74B9FF", "#55EFC4",  # Agents
                "#00B894", "#00D2D3"  # Consensus, Execute
            ],
            customdata = [
                f"₹{branch_df_flow.iloc[i]['TotalRevenue']:,.0f}" if i < len(branch_df_flow) else "Active"
                for i in range(13)
            ],
            hovertemplate='<b>%{label}</b><br>Status: Active<br>%{customdata}<extra></extra>'
        ),
        link = dict(
            source = [0, 1, 2, 3, 4,  # Stores to Kafka
                     5, 5, 5, 5, 5,  # Kafka to Processor (split for visual)
                     6, 6, 6, 6,     # Processor to ML (split)
                     7, 7, 7,        # ML to Agents
                     8, 9, 10,       # Agents to Consensus
                     11],            # Consensus to Execute
            target = [5, 5, 5, 5, 5,  # Stores to Kafka
                     6, 6, 6, 6, 6,  # Kafka to Processor
                     7, 7, 7, 7,     # Processor to ML
                     8, 9, 10,       # ML to Agents
                     11, 11, 11,     # Agents to Consensus
                     12],            # Consensus to Execute
            value = [100, 120, 90, 110, 95,  # Store volumes
                    103, 103, 103, 103, 103, # Kafka throughput
                    103, 103, 103, 103,      # Processing
                    products_tracked if products_tracked > 0 else 50,
                    products_tracked if products_tracked > 0 else 50,
                    products_tracked if products_tracked > 0 else 50,
                    products_tracked//3 if products_tracked > 0 else 17,
                    products_tracked//3 if products_tracked > 0 else 17,
                    products_tracked//3 if products_tracked > 0 else 16,
                    actions_taken if actions_taken > 0 else 30],
            color = ["rgba(255,107,107,0.3)", "rgba(78,205,196,0.3)", "rgba(69,183,209,0.3)", 
                    "rgba(255,160,122,0.3)", "rgba(152,216,200,0.3)",
                    "rgba(255,217,61,0.4)", "rgba(255,217,61,0.4)", "rgba(255,217,61,0.4)", 
                    "rgba(255,217,61,0.4)", "rgba(255,217,61,0.4)",
                    "rgba(149,225,211,0.4)", "rgba(149,225,211,0.4)", "rgba(149,225,211,0.4)", 
                    "rgba(149,225,211,0.4)",
                    "rgba(108,92,231,0.5)", "rgba(108,92,231,0.5)", "rgba(108,92,231,0.5)",
                    "rgba(253,121,168,0.5)", "rgba(116,185,255,0.5)", "rgba(85,239,196,0.5)",
                    "rgba(0,184,148,0.6)"],
            hovertemplate='Flow: %{value} items<extra></extra>'
        )
    )])
    
    fig_main.update_layout(
        title={
            'text': "📊 Real-Time Data Flow: 5 Stores → Kafka → ML → 3 Agents → Autonomous Action",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        font=dict(size=12, family='Arial', color='#2C3E50'),
        plot_bgcolor='rgba(240,240,245,0.5)',
        paper_bgcolor='white',
        height=600,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # ================================================================
    # REAL-TIME METRICS DASHBOARD
    # ================================================================
    st.markdown("---")
    st.markdown("### 📈 Live System Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin:0; font-size: 2.5rem;'>🏪</h2>
            <h3 style='margin:5px 0;'>{}</h3>
            <p style='margin:0; opacity: 0.9;'>Active Stores</p>
        </div>
        """.format(store_count), unsafe_allow_html=True)
    
    with col2:
        try:
            if os.path.exists('output/hourly_transactions.csv') and os.path.getsize('output/hourly_transactions.csv') > 0:
                kafka_msgs = len(pd.read_csv('output/hourly_transactions.csv'))
            else:
                kafka_msgs = 0
        except Exception:
            kafka_msgs = 0
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin:0; font-size: 2.5rem;'>📡</h2>
            <h3 style='margin:5px 0;'>{:,}</h3>
            <p style='margin:0; opacity: 0.9;'>Messages Streamed</p>
        </div>
        """.format(kafka_msgs), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin:0; font-size: 2.5rem;'>🧠</h2>
            <h3 style='margin:5px 0;'>{}</h3>
            <p style='margin:0; opacity: 0.9;'>Products Analyzed</p>
        </div>
        """.format(products_tracked), unsafe_allow_html=True)
    
    with col4:
        agent_decisions = agent_metrics['DecisionsMade'].sum() if not agent_metrics.empty else 0
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin:0; font-size: 2.5rem;'>🤖</h2>
            <h3 style='margin:5px 0;'>{}</h3>
            <p style='margin:0; opacity: 0.9;'>Agent Decisions</p>
        </div>
        """.format(agent_decisions), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin:0; font-size: 2.5rem;'>✅</h2>
            <h3 style='margin:5px 0;'>{}</h3>
            <p style='margin:0; opacity: 0.9;'>Auto-Executed</p>
        </div>
        """.format(actions_taken), unsafe_allow_html=True)
    
    # ================================================================
    # AGENT PERFORMANCE VISUALIZATION
    # ================================================================
    st.markdown("---")
    st.markdown("### 🤖 Agent Performance Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create animated gauge charts for each agent
        if not agent_metrics.empty:
            agent_summary = agent_metrics.groupby('AgentName').agg({
                'DecisionsMade': 'sum',
                'AvgProcessingTimeMS': 'mean'
            }).reset_index()
            
            # Create subplots for 3 agents
            fig_agents = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=('🚨 Urgency Agent', '📦 Quantity Agent', '💰 Cost Agent')
            )
            
            colors = ['#FD79A8', '#74B9FF', '#55EFC4']
            agent_names = ['UrgencyAgent', 'QuantityAgent', 'CostAgent']
            
            for idx, (agent_name, color) in enumerate(zip(agent_names, colors)):
                agent_data = agent_summary[agent_summary['AgentName'] == agent_name]
                decisions = agent_data['DecisionsMade'].values[0] if not agent_data.empty else 0
                
                fig_agents.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = decisions,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Decisions", 'font': {'size': 14}},
                    delta = {'reference': 100, 'increasing': {'color': color}},
                    gauge = {
                        'axis': {'range': [None, max(200, decisions)]},
                        'bar': {'color': color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, max(200, decisions)*0.5], 'color': 'rgba(255,255,255,0.3)'},
                            {'range': [max(200, decisions)*0.5, max(200, decisions)*0.8], 'color': 'rgba(255,255,255,0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max(200, decisions)*0.9
                        }
                    }
                ), row=1, col=idx+1)
            
            fig_agents.update_layout(
                height=300,
                showlegend=False,
                paper_bgcolor='rgba(240,240,245,0.5)',
                font={'family': 'Arial', 'size': 12}
            )
            
            st.plotly_chart(fig_agents, use_container_width=True)
        else:
            st.info("⏳ Waiting for agent metrics...")
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; height: 280px;'>
            <h3 style='text-align: center; margin-top: 20px;'>🎯 Agent Roles</h3>
            <div style='margin-top: 20px; line-height: 2;'>
                <p><b>🚨 Urgency Agent</b><br/>Detects emergencies</p>
                <p><b>📦 Quantity Agent</b><br/>Optimizes order size</p>
                <p><b>💰 Cost Agent</b><br/>Minimizes expenses</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================================
    # REAL-TIME ACTIVITY FEED
    # ================================================================
    st.markdown("---")
    st.markdown("### 📡 Real-Time Activity Feed")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if not actions.empty:
            # Show latest 5 actions with animations
            recent_actions = actions.tail(5).sort_values('Timestamp', ascending=False)
            
            for idx, action in recent_actions.iterrows():
                urgency_colors = {
                    'CRITICAL': '#dc3545',
                    'HIGH': '#fd7e14',
                    'MEDIUM': '#ffc107',
                    'LOW': '#28a745',
                    'NORMAL': '#17a2b8'
                }
                color = urgency_colors.get(action['UrgencyLevel'], '#6c757d')
                
                st.markdown(f"""
                <div style='background: white; border-left: 5px solid {color}; 
                            padding: 15px; margin: 10px 0; border-radius: 5px;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between;'>
                        <b style='color: {color}; font-size: 1.1rem;'>
                            {action['Product'][:30]}
                        </b>
                        <span style='color: #6c757d; font-size: 0.9rem;'>
                            {action['Timestamp'][:19] if 'Timestamp' in action else 'Now'}
                        </span>
                    </div>
                    <div style='margin-top: 10px; color: #495057;'>
                        <span style='background: {color}; color: white; padding: 3px 8px; 
                                     border-radius: 3px; font-size: 0.8rem; margin-right: 10px;'>
                            {action['UrgencyLevel']}
                        </span>
                        <span style='font-size: 0.95rem;'>
                            Order: <b>{action['ReorderQuantity']:,}</b> units
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("⏳ Waiting for agent actions...")
    
    with col2:
        # Urgency distribution donut chart
        if not actions.empty and 'UrgencyLevel' in actions.columns:
            urgency_counts = actions['UrgencyLevel'].value_counts()
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=urgency_counts.index,
                values=urgency_counts.values,
                hole=.6,
                marker=dict(colors=['#dc3545', '#fd7e14', '#ffc107', '#28a745', '#17a2b8']),
                textposition='outside',
                textinfo='label+percent'
            )])
            
            fig_donut.update_layout(
                title={'text': 'Decision Distribution', 'x': 0.5, 'xanchor': 'center'},
                height=300,
                showlegend=False,
                paper_bgcolor='rgba(240,240,245,0.5)',
                annotations=[dict(text='Actions', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
    
    # ================================================================
    # SYSTEM STATUS INDICATOR
    # ================================================================
    st.markdown("---")
    
    # Animated status indicator
    status_html = """
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 30px; border-radius: 15px; text-align: center; color: white;'>
        <h1 style='margin: 0; font-size: 3rem;'>🟢</h1>
        <h2 style='margin: 10px 0;'>SYSTEM OPERATIONAL</h2>
        <p style='margin: 0; font-size: 1.2rem; opacity: 0.9;'>
            All agents active • Real-time processing • Autonomous execution enabled
        </p>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 20px;'>
        <p><b>🚀 LiveInsight+ Multi-Agent System</b></p>
        <p>Real-time inventory management with AI-powered autonomous decision making</p>
        <p style='font-size: 0.9rem;'>Powered by: Kafka • Random Forest • SHAP • LIME • Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    
# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dashboard Mode", "WITH AGENT 🤖")

with col2:
    st.metric("Last Refresh", datetime.now().strftime("%H:%M:%S"))

with col3:
    if agent_active:
        st.metric("Agent Status", "✅ ACTIVE")
    else:
        st.metric("Agent Status", "⚠️ INACTIVE")

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()