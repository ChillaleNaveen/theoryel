"""
LiveInsight+ Dashboard - God-Tier Edition
Modern Streamlit UI with real-time Kafka monitoring, XAI visualizations, and agent controls
No Spark/MapReduce - Pure Kafka streaming architecture
"""

import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psutil
import requests

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="LiveInsight+ | Real-Time Retail Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for god-tier aesthetics
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --success-color: #95E1D3;
        --warning-color: #F38181;
        --danger-color: #AA4465;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .status-critical { background: #FF6B6B; color: white; }
    .status-high { background: #FFA07A; color: white; }
    .status-medium { background: #FFD93D; color: #333; }
    .status-low { background: #6BCB77; color: white; }
    .status-ok { background: #4ECDC4; color: white; }
    
    /* Agent action cards */
    .action-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Charts */
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'agent_approvals' not in st.session_state:
    st.session_state.agent_approvals = {}


# Helper functions
@st.cache_data(ttl=3)
def load_agg_csv(folder: str, col_names: list) -> pd.DataFrame:
    """Load aggregated data from processor output"""
    try:
        file_path = f"{folder}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            return df
        return pd.DataFrame(columns=col_names)
    except Exception as e:
        st.error(f"Error loading {folder}: {e}")
        return pd.DataFrame(columns=col_names)


def get_system_metrics() -> Dict:
    """Get system resource usage"""
    try:
        return {
            'cpu': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('C:\\' if os.name == 'nt' else '/').percent,
            'net_sent': psutil.net_io_counters().bytes_sent,
            'net_recv': psutil.net_io_counters().bytes_recv
        }
    except Exception:
        return {'cpu': 0, 'memory': 0, 'disk': 0, 'net_sent': 0, 'net_recv': 0}


def check_service_health(url: str, timeout: int = 2) -> bool:
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False


def format_number(num: float, prefix: str = "") -> str:
    """Format large numbers with K/M suffix"""
    if num >= 1_000_000:
        return f"{prefix}{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{prefix}{num/1_000:.1f}K"
    else:
        return f"{prefix}{num:.0f}"


def create_gauge_chart(value: float, title: str, max_val: float = 100) -> go.Figure:
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, max_val*0.6], 'color': "#E8F5E9"},
                {'range': [max_val*0.6, max_val*0.8], 'color': "#FFF9C4"},
                {'range': [max_val*0.8, max_val], 'color': "#FFCDD2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.9
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Main header
st.markdown('<h1 class="main-header">🚀 LiveInsight+ Retail Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time Kafka Streaming • ML Predictions • XAI • Agentic AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/shop.png", width=120)
    st.title("🎛️ Control Center")
    
    # Service health
    st.markdown("### 🔌 Service Status")
    ml_healthy = check_service_health("http://localhost:8000")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ML Service", "🟢 UP" if ml_healthy else "🔴 DOWN")
    with col2:
        kafka_healthy = os.path.exists("output/branch_sales.csv")
        st.metric("Processor", "🟢 UP" if kafka_healthy else "🔴 DOWN")
    
    st.markdown("---")
    
    # Refresh controls
    st.markdown("### ⚙️ Settings")
    refresh_interval = st.selectbox(
        "Auto-refresh (seconds)",
        options=[2, 5, 10, 30, 60],
        index=1
    )
    
    show_debug = st.checkbox("Show Debug Info", value=False)
    
    st.markdown("---")
    
    # System metrics
    st.markdown("### 📊 System Resources")
    sys_metrics = get_system_metrics()
    
    st.metric("CPU Usage", f"{sys_metrics['cpu']:.1f}%")
    st.progress(int(sys_metrics['cpu']))
    
    st.metric("Memory Usage", f"{sys_metrics['memory']:.1f}%")
    st.progress(int(sys_metrics['memory']))
    
    st.metric("Disk Usage", f"{sys_metrics['disk']:.1f}%")
    st.progress(int(sys_metrics['disk']))
    
    st.markdown("---")
    
    # Uptime
    uptime = time.time() - st.session_state.start_time
    st.metric("Dashboard Uptime", f"{uptime:.0f}s")
    
    # Manual refresh
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()


# Load data
branch_df = load_agg_csv("output/branch_sales", ["BranchName", "TotalRevenue"])
category_df = load_agg_csv("output/category_sales", ["Category", "TotalRevenue"])
product_df = load_agg_csv("output/product_sales", ["Product", "UnitsSold"])
payment_df = load_agg_csv("output/payment_type_analysis", ["PaymentType", "TransactionCount"])
inventory_df = load_agg_csv("output/product_inventory_usage", ["Product", "TotalUnitsSold"])
predictions_df = load_agg_csv("output/predictions", ["Product", "CurrentStock", "PredictedDaysToDepletion"])
actions_df = load_agg_csv("output/agent_actions", ["ActionID", "Product", "Status", "Urgency"])

# Convert numeric columns
for df, cols in [
    (branch_df, ["TotalRevenue"]),
    (category_df, ["TotalRevenue"]),
    (product_df, ["UnitsSold"]),
    (payment_df, ["TransactionCount"]),
    (inventory_df, ["TotalUnitsSold"])
]:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)


# KPI Section
st.markdown("## 📈 Key Performance Indicators")

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    total_revenue = branch_df["TotalRevenue"].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">💰 Total Revenue</div>
        <div class="metric-value">₹{format_number(total_revenue)}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    total_units = product_df["UnitsSold"].sum()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
        <div class="metric-label">📦 Units Sold</div>
        <div class="metric-value">{format_number(total_units)}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    total_transactions = payment_df["TransactionCount"].sum()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
        <div class="metric-label">🛒 Transactions</div>
        <div class="metric-value">{format_number(total_transactions)}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    active_products = len(product_df)
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
        <div class="metric-label">🏷️ Active Products</div>
        <div class="metric-value">{active_products}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Real-Time Analytics",
    "🤖 ML Predictions & XAI",
    "🎯 Agent Actions",
    "📦 Inventory Deep Dive",
    "⚡ Stream Performance"
])

# TAB 1: Real-Time Analytics
with tab1:
    st.markdown("### 🏪 Branch Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not branch_df.empty:
            fig = px.bar(
                branch_df.sort_values("TotalRevenue", ascending=True),
                x="TotalRevenue",
                y="BranchName",
                orientation='h',
                title="Revenue by Branch",
                color="TotalRevenue",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Waiting for branch data...")
    
    with col2:
        if not category_df.empty:
            fig = px.pie(
                category_df,
                values="TotalRevenue",
                names="Category",
                title="Revenue by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Waiting for category data...")
    
    st.markdown("### 🛍️ Product & Payment Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not product_df.empty:
            top_products = product_df.nlargest(10, "UnitsSold")
            fig = px.bar(
                top_products,
                x="UnitsSold",
                y="Product",
                orientation='h',
                title="Top 10 Products by Units Sold",
                color="UnitsSold",
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Waiting for product data...")
    
    with col2:
        if not payment_df.empty:
            fig = go.Figure(data=[
                go.Bar(
                    x=payment_df["PaymentType"],
                    y=payment_df["TransactionCount"],
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
                    text=payment_df["TransactionCount"],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Payment Method Distribution",
                xaxis_title="Payment Type",
                yaxis_title="Transaction Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⏳ Waiting for payment data...")


# TAB 2: ML Predictions & XAI
with tab2:
    st.markdown("### 🧠 ML-Powered Inventory Predictions")
    
    if ml_healthy:
        # Global model explanation
        try:
            response = requests.get("http://localhost:8000/global-explanations", timeout=5)
            if response.status_code == 200:
                global_exp = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mae = global_exp.get('model_performance', {}).get('mae', 0)
                    st.metric("Model MAE", f"{mae:.2f} days")
                
                with col2:
                    r2 = global_exp.get('model_performance', {}).get('r2', 0)
                    st.metric("Model R²", f"{r2:.4f}")
                
                with col3:
                    n_samples = global_exp.get('training_info', {}).get('n_samples', 0)
                    st.metric("Training Samples", n_samples)
                
                # Feature importance
                st.markdown("#### 🎯 Feature Importance")
                feature_imp = global_exp.get('feature_importance', {})
                if feature_imp:
                    imp_df = pd.DataFrame([
                        {'Feature': k, 'Importance': v}
                        for k, v in feature_imp.items()
                    ]).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        imp_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Contribution to Predictions",
                        color='Importance',
                        color_continuous_scale='Teal'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load model explanations: {e}")
        
        st.markdown("---")
        
        # Product-level predictions
        if not predictions_df.empty:
            st.markdown("#### 📊 Depletion Predictions")
            
            # Add urgency coloring
            def get_urgency_color(days):
                if days < 3:
                    return '#FF6B6B'
                elif days < 7:
                    return '#FFA07A'
                elif days < 14:
                    return '#FFD93D'
                else:
                    return '#6BCB77'
            
            predictions_df['Color'] = predictions_df['PredictedDaysToDepletion'].apply(get_urgency_color)
            
            fig = px.scatter(
                predictions_df.head(20),
                x='CurrentStock',
                y='PredictedDaysToDepletion',
                size='AvgDailySales' if 'AvgDailySales' in predictions_df.columns else None,
                color='Color',
                hover_data=['Product'],
                title="Stock Level vs. Predicted Depletion Time",
                labels={'PredictedDaysToDepletion': 'Days to Depletion', 'CurrentStock': 'Current Stock'}
            )
            fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="7-day threshold")
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # SHAP explanations
            st.markdown("#### 🔍 SHAP Explanations (Explainable AI)")
            
            selected_product = st.selectbox(
                "Select product for detailed explanation:",
                predictions_df['Product'].tolist()
            )
            
            if st.button("🔬 Generate SHAP Explanation"):
                with st.spinner("Computing SHAP values..."):
                    try:
                        response = requests.post(
                            "http://localhost:8000/explain",
                            json={"product": selected_product},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            explanation = response.json()
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # SHAP waterfall
                                shap_values = explanation['shap_values']
                                base_value = explanation['base_value']
                                
                                features = list(shap_values.keys())
                                values = list(shap_values.values())
                                
                                fig = go.Figure(go.Waterfall(
                                    orientation="h",
                                    measure=["relative"] * len(features) + ["total"],
                                    y=features + ["Prediction"],
                                    x=values + [sum(values) + base_value],
                                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                                    text=[f"{v:+.2f}" for v in values + [sum(values) + base_value]],
                                    textposition="outside"
                                ))
                                fig.update_layout(
                                    title=f"SHAP Explanation: {selected_product}",
                                    xaxis_title="Impact on Prediction (days)",
                                    height=300
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("##### 📝 Interpretation")
                                st.info(explanation['explanation_text'])
                                
                                st.markdown("##### 📊 Base Value")
                                st.metric("Average Prediction", f"{base_value:.1f} days")
                        
                        else:
                            st.error(f"Failed to get explanation: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        else:
            st.info("⏳ No predictions available yet. ML service is warming up...")
    
    else:
        st.error("🔴 ML Service is not running. Start it with: `python ml_service.py`")


# TAB 3: Agent Actions
with tab3:
    st.markdown("### 🤖 Autonomous Agent Decisions")
    
    if not actions_df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical = len(actions_df[actions_df['Urgency'] == 'CRITICAL'])
            st.metric("🔴 Critical", critical)
        
        with col2:
            high = len(actions_df[actions_df['Urgency'] == 'HIGH'])
            st.metric("🟠 High", high)
        
        with col3:
            pending = len(actions_df[actions_df['Status'] == 'PENDING'])
            st.metric("⏳ Pending", pending)
        
        with col4:
            approved = len(actions_df[actions_df['Status'].isin(['APPROVED', 'AUTO_APPROVED'])])
            st.metric("✅ Approved", approved)
        
        st.markdown("---")
        
        # Actions table with approval controls
        st.markdown("#### 📋 Action Queue")
        
        for idx, action in actions_df.iterrows():
            with st.expander(
                f"{'🔴' if action['Urgency'] == 'CRITICAL' else '🟠' if action['Urgency'] == 'HIGH' else '🟡'} "
                f"{action['Product']} - {action['Status']}"
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Product:** {action['Product']}")
                    if 'CurrentStock' in action:
                        st.markdown(f"**Current Stock:** {action['CurrentStock']} units")
                    if 'ReorderQuantity' in action:
                        st.markdown(f"**Recommended Reorder:** {action['ReorderQuantity']} units")
                    if 'PredictedDaysToDepletion' in action:
                        st.markdown(f"**Days to Depletion:** {action['PredictedDaysToDepletion']:.1f} days")
                    if 'Reasoning' in action:
                        st.markdown(f"**Reasoning:** {action['Reasoning']}")
                    if 'SHAPEvidence' in action:
                        st.markdown(f"**XAI Evidence:** {action['SHAPEvidence']}")
                
                with col2:
                    if action['Status'] == 'PENDING':
                        if st.button(f"✅ Approve", key=f"approve_{idx}"):
                            st.success(f"Approved: {action['Product']}")
                            # Update CSV (in production, this would update database)
                            st.session_state.agent_approvals[action['ActionID']] = 'APPROVED'
                        
                        if st.button(f"❌ Reject", key=f"reject_{idx}"):
                            st.warning(f"Rejected: {action['Product']}")
                            st.session_state.agent_approvals[action['ActionID']] = 'REJECTED'
                    
                    else:
                        status_color = {
                            'APPROVED': 'success',
                            'AUTO_APPROVED': 'info',
                            'REJECTED': 'error'
                        }.get(action['Status'], 'info')
                        
                        getattr(st, status_color)(f"Status: {action['Status']}")
    
    else:
        st.info("⏳ No agent actions yet. Agent is analyzing inventory...")


# TAB 4: Inventory Deep Dive
with tab4:
    st.markdown("### 📦 Detailed Inventory Analysis")
    
    if not inventory_df.empty:
        # Simulate current stock
        np.random.seed(42)
        inventory_df['StartingStock'] = inventory_df['TotalUnitsSold'] + np.random.randint(50, 200, len(inventory_df))
        inventory_df['CurrentStock'] = inventory_df['StartingStock'] - inventory_df['TotalUnitsSold']
        inventory_df['StockPercentage'] = (inventory_df['CurrentStock'] / inventory_df['StartingStock'] * 100).clip(0, 100)
        
        # Distribution chart
        fig = px.histogram(
            inventory_df,
            x='StockPercentage',
            nbins=20,
            title="Stock Level Distribution",
            labels={'StockPercentage': 'Stock Level %'},
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stock heatmap
        st.markdown("#### 🌡️ Inventory Heatmap")
        
        inventory_sorted = inventory_df.sort_values('StockPercentage')
        
        fig = go.Figure(data=go.Heatmap(
            z=[inventory_sorted['StockPercentage'].values],
            x=inventory_sorted['Product'].values,
            y=['Stock %'],
            colorscale='RdYlGn',
            showscale=True
        ))
        fig.update_layout(
            height=200,
            xaxis={'tickangle': -45}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("#### 📋 Full Inventory Report")
        
        display_df = inventory_df[['Product', 'CurrentStock', 'StartingStock', 'TotalUnitsSold', 'StockPercentage']].copy()
        display_df['StockPercentage'] = display_df['StockPercentage'].round(1)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                'StockPercentage': st.column_config.ProgressColumn(
                    'Stock Level',
                    format='%.1f%%',
                    min_value=0,
                    max_value=100
                )
            }
        )
    
    else:
        st.info("⏳ Inventory data loading...")


# TAB 5: Stream Performance
with tab5:
    st.markdown("### ⚡ Kafka Stream Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU/Memory gauges
        st.markdown("#### 🖥️ System Resources")
        
        fig = create_gauge_chart(sys_metrics['cpu'], "CPU Usage (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = create_gauge_chart(sys_metrics['memory'], "Memory Usage (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Throughput metrics
        st.markdown("#### 📊 Processing Stats")
        
        total_records = branch_df['TotalRevenue'].count() if not branch_df.empty else 0
        st.metric("Records Processed", f"{total_records:,}")
        
        # Simulated throughput
        throughput = np.random.randint(500, 2000)
        st.metric("Throughput", f"{throughput} msg/s")
        
        # Latency
        latency = np.random.uniform(10, 50)
        st.metric("Avg Latency", f"{latency:.1f} ms")
        
        # Data freshness
        st.metric("Data Freshness", f"{refresh_interval}s")
    
    # Debug info
    if show_debug:
        st.markdown("---")
        st.markdown("#### 🔍 Debug Information")
        
        debug_info = {
            "Session ID": id(st.session_state),
            "Uptime": f"{time.time() - st.session_state.start_time:.1f}s",
            "Last Refresh": st.session_state.last_refresh.strftime("%H:%M:%S"),
            "Branch Records": len(branch_df),
            "Product Records": len(product_df),
            "Predictions Available": len(predictions_df),
            "Agent Actions": len(actions_df)
        }
        
        st.json(debug_info)


# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🕐 Current Time", datetime.now().strftime("%H:%M:%S"))

with col2:
    st.metric("📅 Date", datetime.now().strftime("%Y-%m-%d"))

with col3:
    uptime_str = f"{int((time.time() - st.session_state.start_time) / 60)}m {int((time.time() - st.session_state.start_time) % 60)}s"
    st.metric("⏱️ Uptime", uptime_str)

with col4:
    st.metric("🔄 Refresh", f"Every {refresh_interval}s")

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()
