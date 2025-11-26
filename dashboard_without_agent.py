"""
LiveInsight Dashboard WITHOUT Autonomous Agent
Shows real-time retail analytics with manual inventory management
Demonstrates traditional approach without agentic AI
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

# Streamlit Configuration
st.set_page_config(
    page_title="LiveInsight WITHOUT Agent",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
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
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #dc3545;
        text-align: center;
        margin-bottom: 1rem;
    }
    .manual-mode {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
# Replace your existing load_csv function with this:
@st.cache_data(ttl=3)
def load_csv(filepath, columns=None):
    """Load CSV safely, handling missing or empty files"""
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            df = pd.read_csv(filepath)
            if columns:
                # Ensure we only select columns that actually exist
                valid_cols = [c for c in columns if c in df.columns]
                if valid_cols:
                    df = df[valid_cols]
            return df
        return pd.DataFrame(columns=columns if columns else [])
    except Exception as e:
        # st.error(f"Error loading {filepath}: {e}") # Optional: comment out to hide errors
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
st.sidebar.title("📊 LiveInsight WITHOUT Agent")
st.sidebar.markdown("---")

refresh_interval = st.sidebar.selectbox(
    "Auto-refresh interval (seconds)",
    options=[2, 5, 10, 30], 
    index=1
)

# Manual Mode Indicator
st.sidebar.warning("📋 **MANUAL MODE**")
st.sidebar.markdown("All decisions require human review")

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

inventory_df = load_csv("output/product_inventory_usage.csv", ["Product", "TotalUnitsSold"])
predictions_df = load_csv("output/predictions.csv")

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

# Network Stats
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Network I/O")
st.sidebar.metric("Data Sent", f"{net_sent_mb:.1f} MB")
st.sidebar.metric("Data Received", f"{net_recv_mb:.1f} MB")

# Main Dashboard
st.markdown('<div class="main-header">📊 LiveInsight Retail Intelligence WITHOUT Autonomous Agent</div>', 
            unsafe_allow_html=True)

# Manual Mode Banner
st.markdown("""
<div class="manual-mode">
    <h3>📋 Manual Inventory Management Mode</h3>
    <p>This system operates WITHOUT autonomous agents. All inventory decisions require manual human review and approval.
    You must actively monitor dashboards, identify issues, and create reorder requests manually.</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Business Analytics", 
    "🔮 ML Predictions (Manual Review)", 
    "📦 Inventory Status",
    "⚠️ Manual Alert Center"
])

# Tab 1: Business Analytics (Same as agent version)
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

# Tab 2: ML Predictions (Manual Review Required)
with tab2:
    st.header("🔮 Machine Learning Predictions - Manual Review Required")
    
    st.warning("""
    ⚠️ **Manual Review Required**: These predictions require human analysis and decision-making.
    You must review each product, assess urgency, and manually create reorder requests.
    """)
    
    if predictions_df.empty:
        st.warning("⚠️ No predictions available. ML service may not be running.")
        st.info("Start ML service with: `python ml_service_enhanced.py`")
    else:
        st.info(f"📊 {len(predictions_df)} products require manual review")
        
        # Calculate urgency based on DaysToDepletion
        if 'DaysToDepletion' in predictions_df.columns:
            def classify_urgency(days):
                try:
                    days = float(days)
                    if days < 3:
                        return 'CRITICAL'
                    elif days < 7:
                        return 'HIGH'
                    elif days < 14:
                        return 'MEDIUM'
                    elif days < 30:
                        return 'LOW'
                    else:
                        return 'NORMAL'
                except:
                    return 'UNKNOWN'
            
            predictions_df['UrgencyLevel'] = predictions_df['DaysToDepletion'].apply(classify_urgency)
        else:
            predictions_df['UrgencyLevel'] = 'UNKNOWN'
        
        # Urgency distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Products by Urgency")
            urgency_counts = predictions_df['UrgencyLevel'].value_counts()
            fig = px.pie(
                values=urgency_counts.values,
                names=urgency_counts.index,
                title="Urgency Level Distribution (Needs Manual Action)",
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, width='stretch')
            
            # Show counts
            st.markdown("### Manual Actions Needed:")
            for level, count in urgency_counts.items():
                if level == 'CRITICAL':
                    st.error(f"🔴 **{level}**: {count} products - IMMEDIATE ACTION REQUIRED!")
                elif level == 'HIGH':
                    st.warning(f"🟠 **{level}**: {count} products - Review within 24 hours")
                elif level == 'MEDIUM':
                    st.info(f"🟡 **{level}**: {count} products - Review this week")
                else:
                    st.success(f"🟢 **{level}**: {count} products - Monitor regularly")
        
        with col2:
            st.subheader("📊 Time to Review")
            
            # Calculate estimated manual review time
            total_products = len(predictions_df)
            time_per_review = 5  # minutes per product
            total_time = total_products * time_per_review
            
            st.metric("Products to Review", total_products)
            st.metric("Estimated Time", f"{total_time} minutes", 
                     help=f"Based on {time_per_review} min per product")
            st.metric("If 2 people", f"{total_time/2:.0f} minutes")
            
            st.markdown("""
            **Manual Review Process:**
            1. Review each product individually
            2. Check current stock levels
            3. Analyze sales trends
            4. Determine reorder quantity
            5. Create purchase order
            6. Get manager approval
            7. Submit to supplier
            
            *This process repeats for every product!*
            """)
        
        st.markdown("---")
        
        # Detailed predictions table for manual review
        st.subheader("📋 Detailed Predictions - Manual Review")
        
        # Filter by urgency
        urgency_filter = st.selectbox(
            "Filter by Urgency Level",
            ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]
        )
        
        display_df = predictions_df.copy()
        if urgency_filter != "All":
            display_df = display_df[display_df['UrgencyLevel'] == urgency_filter]
        
        # Color coding
        def highlight_urgency(row):
            if row['UrgencyLevel'] == 'CRITICAL':
                return ['background-color: #ffcccc'] * len(row)
            elif row['UrgencyLevel'] == 'HIGH':
                return ['background-color: #ffe6cc'] * len(row)
            elif row['UrgencyLevel'] == 'MEDIUM':
                return ['background-color: #ffffcc'] * len(row)
            return [''] * len(row)
        
        display_cols = ['Product', 'CurrentStock', 'AvgDailySales', 
                       'PredictedDaysToDepletion', 'Confidence', 'UrgencyLevel', 'Recommendation']
        
        st.dataframe(
            display_df.style.apply(highlight_urgency, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Manual action form
        st.markdown("---")
        st.subheader("📝 Manual Reorder Request Form")
        
        with st.form("manual_reorder_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                product_select = st.selectbox(
                    "Select Product",
                    display_df['Product'].tolist() if not display_df.empty else []
                )
                
                reorder_qty = st.number_input(
                    "Reorder Quantity",
                    min_value=0,
                    value=100,
                    step=12,
                    help="Calculate manually based on sales velocity"
                )
            
            with col2:
                urgency = st.selectbox(
                    "Set Urgency",
                    ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                )
                
                notes = st.text_area(
                    "Notes / Justification",
                    help="Explain why this reorder is needed"
                )
            
            submitted = st.form_submit_button("Submit for Manager Approval")
            
            if submitted:
                st.info(f"""
                📋 **Reorder Request Submitted** (Demo - not persisted)
                - Product: {product_select}
                - Quantity: {reorder_qty} units
                - Urgency: {urgency}
                - Status: Pending Manager Approval
                
                ⏳ This request will now wait in queue for manager review and approval.
                Typical approval time: 2-24 hours depending on manager availability.
                """)

# Tab 3: Inventory Status
with tab3:
    st.header("📦 Current Inventory Status - Manual Monitoring")
    
    st.warning("""
    ⚠️ **Manual Monitoring**: You must actively check this page regularly to identify stock issues.
    No automatic alerts or notifications are available.
    """)
    
    if not inventory_df.empty and not predictions_df.empty:
        # Merge inventory with predictions
        merged_df = inventory_df.merge(predictions_df, on='Product', how='left')
        
        # Stock level visualization
        st.subheader("📊 Stock Levels Overview")
        
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
                    st.error(f"🔴 {status}: {count} products - **CHECK NOW!**")
                elif status == 'Low':
                    st.warning(f"🟠 {status}: {count} products - **Review soon**")
                elif status == 'Medium':
                    st.info(f"🟡 {status}: {count} products - **Monitor**")
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
                title="Products by Stock Status (Manual Action Required)",
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
        
        # Critical items alert
        critical_items = merged_df[merged_df['StockStatus'] == 'Critical']
        if not critical_items.empty:
            st.error(f"""
            🚨 **CRITICAL ALERT**: {len(critical_items)} products at critical stock levels!
            
            These items will stockout in less than 3 days. Manual intervention required IMMEDIATELY!
            """)
            
            st.dataframe(
                critical_items[['Product', 'CurrentStock', 'AvgDailySales', 
                               'PredictedDaysToDepletion']],
                use_container_width=True,
                
            )
        
        # Detailed inventory table
        st.subheader("📋 Detailed Inventory Report (Manual Review)")
        display_df = merged_df[['Product', 'TotalUnitsSold', 'CurrentStock', 
                                'AvgDailySales', 'PredictedDaysToDepletion', 
                                'StockStatus']].sort_values('PredictedDaysToDepletion')
        
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("Inventory data will appear once streaming data is processed")

# Tab 4: Manual Alert Center
with tab4:
    st.header("⚠️ Manual Alert Center")
    
    st.markdown("""
    ### 📋 Manual Checklist
    
    Without an autonomous agent, you must manually:
    """)
    
    st.markdown("""
    - [ ] Check dashboards multiple times per day
    - [ ] Review all ML predictions individually
    - [ ] Identify critical and high-priority items
    - [ ] Calculate appropriate reorder quantities
    - [ ] Create reorder requests for each product
    - [ ] Wait for manager approval (2-24 hours)
    - [ ] Follow up on pending approvals
    - [ ] Submit approved orders to suppliers
    - [ ] Track order status manually
    - [ ] Update inventory records after delivery
    """)
    
    st.markdown("---")
    
    # Manual workload estimation
    st.subheader("📊 Manual Workload Estimate")
    
    if not predictions_df.empty:
        urgent_items = len(predictions_df[predictions_df['UrgencyLevel'].isin(['CRITICAL', 'HIGH'])])
        total_items = len(predictions_df)
        
        # Time estimates (minutes)
        review_time = total_items * 5
        decision_time = urgent_items * 10
        paperwork_time = urgent_items * 15
        total_time = review_time + decision_time + paperwork_time
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Items Needing Review", total_items)
            st.caption("All products require manual attention")
        
        with col2:
            st.metric("Urgent Items", urgent_items)
            st.caption("Require immediate action")
        
        with col3:
            st.metric("Estimated Work Time", f"{total_time} min")
            st.caption(f"≈ {total_time/60:.1f} hours of manual work")
        
        st.markdown("---")
        
        # Breakdown
        st.markdown("### ⏱️ Time Breakdown")
        
        breakdown_df = pd.DataFrame({
            'Task': ['Review Predictions', 'Make Decisions', 'Create Paperwork'],
            'Time (minutes)': [review_time, decision_time, paperwork_time]
        })
        
        fig = px.bar(
            breakdown_df,
            x='Task',
            y='Time (minutes)',
            title="Manual Processing Time by Task",
            color='Time (minutes)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, width='stretch')
        
        st.warning(f"""
        ⚠️ **Workload Alert**: Current inventory situation requires approximately **{total_time/60:.1f} hours** 
        of manual analysis and decision-making. This does not include waiting time for approvals or 
        supplier communication.
        
        With an autonomous agent, this would be reduced to minutes with automatic recommendations and approvals.
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dashboard Mode", "WITHOUT AGENT 📋")

with col2:
    st.metric("Last Refresh", datetime.now().strftime("%H:%M:%S"))

with col3:
    st.metric("Automation Level", "0% (Manual)")

st.info("""
💡 **Want to see the difference?** Open the agent-powered dashboard side-by-side:
- This dashboard: Manual inventory management (human-dependent)
- Agent dashboard: Autonomous inventory management (AI-powered)

Run both and compare the efficiency!
""")

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()

