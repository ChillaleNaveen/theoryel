# dashboard.py
import streamlit as st
import pandas as pd
import glob
import os
import time
import psutil
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.figure_factory as ff
import threading
import queue
from contextlib import contextmanager

# WMI Setup with proper COM initialization
wmi_available = False
wmi_queue = queue.Queue()

def init_wmi():
    """Initialize WMI in a separate thread with proper COM initialization"""
    global wmi_available
    try:
        import pythoncom
        import wmi
        
        # Initialize COM for this thread
        pythoncom.CoInitialize()
        
        # Create WMI connection
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        wmi_available = True
        return w
    except ImportError:
        print("WMI not available - Install with: pip install wmi pywin32")
        return None
    except Exception as e:
        print(f"WMI initialization failed: {e}")
        return None

def get_wmi_temperatures():
    """Get temperatures from WMI in a thread-safe way"""
    if not wmi_available:
        return None, None
    
    try:
        import pythoncom
        import wmi
        
        # Initialize COM for this thread
        pythoncom.CoInitialize()
        
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()
        
        cpu_temp = None
        gpu_temp = None
        
        for sensor in sensors:
            if sensor.SensorType == "Temperature":
                if "CPU" in sensor.Name or "Package" in sensor.Name:
                    cpu_temp = sensor.Value
                elif "GPU" in sensor.Name:
                    gpu_temp = sensor.Value
                    
        pythoncom.CoUninitialize()
        return cpu_temp, gpu_temp
        
    except Exception as e:
        st.sidebar.error(f"WMI Error: {e}")
        return None, None

@contextmanager
def wmi_context():
    """Context manager for WMI operations"""
    try:
        import pythoncom
        pythoncom.CoInitialize()
        yield
    except ImportError:
        yield
    finally:
        try:
            pythoncom.CoUninitialize()
        except:
            pass

# Streamlit Configuration
st.set_page_config(
    page_title="LiveInsight Retail Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data(ttl=5)
def load_agg_csv(folder, col_names):
    """Load and aggregate CSV data from Spark output"""
    path = os.path.join(folder, "part-*.csv")
    files = glob.glob(path)
    
    if not files:
        return pd.DataFrame(columns=col_names)
    
    df_list = []
    for f in files:
        try:
            if os.path.getsize(f) > 0:  # Check if file is not empty
                df_part = pd.read_csv(f, header=None, names=col_names)
                if not df_part.empty:
                    df_list.append(df_part)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        except Exception as e:
            st.sidebar.warning(f"Error reading {f}: {e}")
            continue
    
    if not df_list:
        return pd.DataFrame(columns=col_names)
    
    return pd.concat(df_list, ignore_index=True)

def get_system_temperatures():
    """Get system temperatures from multiple sources"""
    cpu_temp = None
    gpu_temp = None
    
    # Try psutil first (most reliable)
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            
            # Try different sensor names for CPU
            if "coretemp" in temps:
                for sensor in temps["coretemp"]:
                    if "Package id 0" in sensor.label:
                        cpu_temp = sensor.current
                        break
                    elif "Core 0" in sensor.label and cpu_temp is None:
                        cpu_temp = sensor.current
            
            # Try different sensor names for GPU
            if "nvidia" in temps:
                gpu_temp = temps["nvidia"][0].current
            elif "amdgpu" in temps:
                gpu_temp = temps["amdgpu"][0].current
                
        except Exception as e:
            st.sidebar.warning(f"psutil temperature error: {e}")
    
    # Try WMI as fallback (Windows only)
    if cpu_temp is None or gpu_temp is None:
        try:
            wmi_cpu, wmi_gpu = get_wmi_temperatures()
            if cpu_temp is None:
                cpu_temp = wmi_cpu
            if gpu_temp is None:
                gpu_temp = wmi_gpu
        except Exception as e:
            st.sidebar.warning(f"WMI temperature error: {e}")
    
    # Fallback to simulation if no real data
    return (
    float(cpu_temp) if cpu_temp is not None else None,
    float(gpu_temp) if gpu_temp is not None else None
)


def create_sample_data():
    """Create sample data if CSV files don't exist"""
    branches = ["Mumbai Central", "Delhi North", "Bangalore South", "Chennai East", "Kolkata West"]
    categories = ["Electronics", "Clothing", "Groceries", "Books", "Sports"]
    products = ["Laptop", "Smartphone", "T-Shirt", "Jeans", "Rice", "Milk", "Novel", "Football"]
    payments = ["Credit Card", "Debit Card", "Cash", "UPI", "Net Banking"]
    
    branch_data = pd.DataFrame({
        "BranchName": branches,
        "TotalRevenue": np.random.uniform(50000, 200000, len(branches))
    })
    
    cat_data = pd.DataFrame({
        "Category": categories,
        "TotalRevenue": np.random.uniform(30000, 150000, len(categories))
    })
    
    prod_data = pd.DataFrame({
        "Product": products,
        "UnitsSold": np.random.randint(100, 1000, len(products))
    })
    
    payment_data = pd.DataFrame({
        "PaymentType": payments,
        "TransactionCount": np.random.randint(500, 2000, len(payments))
    })
    
    return branch_data, cat_data, prod_data, payment_data

# --- Initialize WMI ---
try:
    with wmi_context():
        wmi_w = init_wmi()
except Exception as e:
    st.sidebar.info(f"WMI initialization: {e}")

# --- Sidebar Controls ---
st.sidebar.title("🔧 LiveInsight Controls")

refresh_interval = st.sidebar.selectbox(
    "Auto-refresh interval (seconds)",
    options=[2, 5, 10, 30], 
    index=1
)

# Demo mode toggle
demo_mode = st.sidebar.checkbox("Demo Mode (Use Sample Data)", value=False)
# Display last refresh time
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown(f"**Last Refresh:** {datetime.now().strftime('%H:%M:%S')}")

# --- System Resource Usage ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Resources")

try:
    cpu_usage = psutil.cpu_percent(interval=0.1)
    mem_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent
    
    st.sidebar.progress(int(cpu_usage), text=f"CPU: {cpu_usage:.1f}%")
    st.sidebar.progress(int(mem_usage), text=f"Memory: {mem_usage:.1f}%")
    st.sidebar.progress(int(disk_usage), text=f"Disk: {disk_usage:.1f}%")
    
    # Network stats
    net_io = psutil.net_io_counters()
    
except Exception as e:
    st.sidebar.error(f"System monitoring error: {e}")

# --- Load or Generate Data ---
if demo_mode:
    branch_df, cat_df, prod_df, payment_df = create_sample_data()
    weekly_df = pd.DataFrame(columns=["BranchName", "Week", "TotalRevenue"])
    monthly_df = pd.DataFrame(columns=["BranchName", "Month", "TotalRevenue"])
    st.sidebar.success("Using sample data")
else:
    branch_df = load_agg_csv("output/branch_sales", ["BranchName", "TotalRevenue"])
    branch_df["TotalRevenue"] = pd.to_numeric(branch_df["TotalRevenue"], errors="coerce")
    branch_df = branch_df[branch_df["BranchName"] != "BranchName"]

    cat_df = load_agg_csv("output/category_sales", ["Category", "TotalRevenue"])
    cat_df["TotalRevenue"] = pd.to_numeric(cat_df["TotalRevenue"], errors="coerce")

    prod_df = load_agg_csv("output/product_sales", ["Product", "UnitsSold"])
    prod_df["UnitsSold"] = pd.to_numeric(prod_df["UnitsSold"], errors="coerce")

    payment_df = load_agg_csv("output/payment_type_analysis", ["PaymentType", "TransactionCount"])
    payment_df["TransactionCount"] = pd.to_numeric(payment_df["TransactionCount"], errors="coerce")

    weekly_df = load_agg_csv("output/weekly_branch_revenue", ["BranchName", "Week", "TotalRevenue"])
    weekly_df["TotalRevenue"] = pd.to_numeric(weekly_df["TotalRevenue"], errors="coerce")

    monthly_df = load_agg_csv("output/monthly_branch_revenue", ["BranchName", "Month", "TotalRevenue"])
    monthly_df["TotalRevenue"] = pd.to_numeric(monthly_df["TotalRevenue"], errors="coerce")




# --- KPIs ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Key Metrics")

total_revenue = branch_df["TotalRevenue"].sum() if not branch_df.empty else 0
total_units = prod_df["UnitsSold"].sum() if not prod_df.empty else 0
num_branches = branch_df.shape[0] if not branch_df.empty else 0
if not payment_df.empty:
    payment_df['TransactionCount'] = (
        payment_df['TransactionCount']
        .astype(str)
        .str.extract('(\d+)')  # extract numeric part
        .astype(float)
    )
    avg_transaction = payment_df['TransactionCount'].mean()
else:
    avg_transaction = 0

st.sidebar.metric("Total Revenue", f"₹{total_revenue:,.0f}")
st.sidebar.metric("Total Units Sold", f"{total_units:,}")
st.sidebar.metric("Active Branches", f"{num_branches}")
st.sidebar.metric("Avg Transactions", f"{avg_transaction:.0f}")

# --- Main Dashboard ---
st.title("🚀 LiveInsight: Retail Heat & System Monitor")
st.markdown("*Real-time business intelligence with system performance monitoring*")

# --- Temperature Monitoring ---
st.header("🌡️ System Temperature Monitoring")

# Initialize session state for temperature history
if "temp_history" not in st.session_state:
    st.session_state.temp_history = {
        "timestamps": [],
        "cpu_temps": [],
        "gpu_temps": [],
        "cpu_usage": [],
        "memory_usage": []
    }

# Get current temperatures and system stats
try:
    cpu_temp, gpu_temp = get_system_temperatures()
    current_cpu = psutil.cpu_percent(interval=0.1)
    current_mem = psutil.virtual_memory().percent
    
    # Update history
    current_time = datetime.now()
    st.session_state.temp_history["timestamps"].append(current_time)
    st.session_state.temp_history["cpu_temps"].append(cpu_temp)
    st.session_state.temp_history["gpu_temps"].append(gpu_temp)
    st.session_state.temp_history["cpu_usage"].append(current_cpu)
    st.session_state.temp_history["memory_usage"].append(current_mem)
    
    # Keep only last 50 readings
    for key in st.session_state.temp_history:
        st.session_state.temp_history[key] = st.session_state.temp_history[key][-50:]
    
    # Display current temperatures
    col1, col2, col3, col4 = st.columns(4)
    

    with col1:
        if cpu_temp is not None:
            temp_color = "normal" if cpu_temp < 70 else "inverse"
            st.metric("CPU Temperature", f"{cpu_temp:.1f}°C", delta=None, delta_color=temp_color)
        else:
            st.metric("CPU Temperature", "Can't Detect")

    with col2:
        if gpu_temp is not None:
            temp_color = "normal" if gpu_temp < 75 else "inverse"
            st.metric("GPU Temperature", f"{gpu_temp:.1f}°C", delta=None, delta_color=temp_color)
        else:
            st.metric("GPU Temperature", "Can't Detect")

    
    
    # Temperature trend chart
    if len(st.session_state.temp_history["timestamps"]) > 1:
        temp_df = pd.DataFrame({
            "Time": st.session_state.temp_history["timestamps"],
            "CPU Temp (°C)": st.session_state.temp_history["cpu_temps"],
            "GPU Temp (°C)": st.session_state.temp_history["gpu_temps"],
        })
        
        st.subheader("📊 System Performance Trends")
        st.line_chart(temp_df.set_index("Time"))
        
        # Temperature alerts
        if cpu_temp is not None:
            if cpu_temp > 80:
                st.error(f"⚠️ HIGH CPU TEMPERATURE: {cpu_temp:.1f}°C")
            elif cpu_temp > 70:
                st.warning(f"🔥 Elevated CPU temperature: {cpu_temp:.1f}°C")

        if gpu_temp is not None:
            if gpu_temp > 85:
                st.error(f"⚠️ HIGH GPU TEMPERATURE: {gpu_temp:.1f}°C")
            elif gpu_temp > 75:
                st.warning(f"🔥 Elevated GPU temperature: {gpu_temp:.1f}°C")


except Exception as e:
    st.error(f"Temperature monitoring error: {e}")

# --- Main Retail Dashboard ---
st.header("🏪 Retail Performance Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Branch Performance")
    if not branch_df.empty:
        fig = px.bar(
            branch_df.sort_values("TotalRevenue", ascending=False),
            x="BranchName", 
            y="TotalRevenue",
            color="TotalRevenue",
            color_continuous_scale="viridis",
            title="Revenue by Branch"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Branch Details"):
            cleaned_branch_df = branch_df[branch_df["BranchName"] != "BranchName"]
            st.dataframe(cleaned_branch_df.sort_values("TotalRevenue", ascending=False), use_container_width=True)

    else:
        st.info("📊 No branch data available")

    st.subheader("🏷️ Category Performance")
    if not cat_df.empty:
        fig = px.pie(
            cat_df, 
            names="Category", 
            values="TotalRevenue",
            title="Revenue Distribution by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Category Details"):
            st.dataframe(cat_df.sort_values("TotalRevenue", ascending=False), use_container_width=True)
    else:
        st.info("📊 No category data available")

with col2:
    st.subheader("🛍️ Top Products")
    if not prod_df.empty:
        top_products = prod_df.sort_values("UnitsSold", ascending=False).head(10)
        fig = px.bar(
            top_products, 
            y="Product", 
            x="UnitsSold", 
            orientation='h',
            color="UnitsSold",
            color_continuous_scale="plasma",
            title="Top 10 Products by Units Sold"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Product Details"):
            st.dataframe(top_products, use_container_width=True)
    else:
        st.info("📊 No product data available")

    st.subheader("💳 Payment Methods")
    if not payment_df.empty:
        fig = px.pie(
            payment_df, 
            names="PaymentType", 
            values="TransactionCount",
            title="Payment Method Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4  # This creates a donut chart effect
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Payment Details"):
            st.dataframe(payment_df.sort_values("TransactionCount", ascending=False), use_container_width=True)
    else:
        st.info("📊 No payment data available")

st.header("📆 Time-Based Branch Revenue Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Weekly Revenue Trend")
    if not weekly_df.empty:
        fig = px.line(
            weekly_df.sort_values(["BranchName", "Week"]),
            x="Week", y="TotalRevenue", color="BranchName",
            title="Branch Revenue Over Weeks",
            markers=True
        )
        fig.update_layout(height=400, xaxis_title="Week", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Weekly Revenue Data"):
            st.dataframe(weekly_df.sort_values(["BranchName", "Week"]), use_container_width=True)
    else:
        st.info("📊 No weekly revenue data available")

with col2:
    st.subheader("📆 Monthly Revenue Trend")
    if not monthly_df.empty:
        fig = px.line(
            monthly_df.sort_values(["BranchName", "Month"]),
            x="Month", y="TotalRevenue", color="BranchName",
            title="Branch Revenue Over Months",
            markers=True
        )
        fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Revenue")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Monthly Revenue Data"):
            st.dataframe(monthly_df.sort_values(["BranchName", "Month"]), use_container_width=True)
    else:
        st.info("📊 No monthly revenue data available")


# --- Inventory Management Section ---
# --- ML & Data Processing Imports for Inventory ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# --- Inventory Management Section (with ML Prediction over 5 Days) ---
st.header("📦 Predictive Inventory Management")

# Define the sales period. This makes it easy to change later.
SALES_PERIOD_DAYS = 5

# --- Helper Function to Train a Predictive Model ---
@st.cache_resource
def train_depletion_model(df):
    """
    Simulates historical sales data over a short period (5 days) and trains a 
    RandomForestRegressor model to predict days to depletion.
    """
    # 1. Simulate historical daily sales data for training
    historical_data = []
    for _, row in df.iterrows():
        product = row['Product']
        total_sales = row['TotalUnitsSold']
        if total_sales <= 0:
            continue
        
        # Simulate 5 days of sales history based on the total
        avg_daily_sale = total_sales / SALES_PERIOD_DAYS
        # Add random noise for realism
        daily_sales = np.random.normal(loc=avg_daily_sale, scale=avg_daily_sale * 0.3, size=SALES_PERIOD_DAYS).astype(int)
        daily_sales[daily_sales < 0] = 0
        
        current_stock = row['StartingStock']
        for i in range(SALES_PERIOD_DAYS):
            stock_at_start_of_day = current_stock
            sales_today = daily_sales[i]
            if stock_at_start_of_day <= 0:
                break
                
            features = {
                'Product': product,
                'CurrentStock': stock_at_start_of_day,
                'AvgDailySales': avg_daily_sale
            }
            
            # Target: How many days until stock runs out from this point?
            remaining_stock = stock_at_start_of_day
            days_to_deplete = 0
            if avg_daily_sale > 0:
                days_to_deplete = np.ceil(remaining_stock / avg_daily_sale)

            features['DaysToDepletion'] = days_to_deplete
            historical_data.append(features)
            
            current_stock -= sales_today

    if not historical_data:
        return None

    train_df = pd.DataFrame(historical_data)
    
    # 2. Train the Random Forest Model
    X = train_df[['CurrentStock', 'AvgDailySales']]
    y = train_df['DaysToDepletion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    st.sidebar.info(f"ML Model Accuracy (R²): {model.score(X_test, y_test):.2f}")
    
    return model

# --- Main Inventory Logic ---
inventory_df = load_agg_csv("output/product_inventory_usage", ["Product", "TotalUnitsSold"])
inventory_df["TotalUnitsSold"] = pd.to_numeric(inventory_df["TotalUnitsSold"], errors='coerce').fillna(0).astype(int)

np.random.seed(42)
inventory_df["StartingStock"] = inventory_df["TotalUnitsSold"] + np.random.randint(50, 150, size=len(inventory_df))
inventory_df["CurrentStock"] = inventory_df["StartingStock"] - inventory_df["TotalUnitsSold"]
# **KEY CHANGE**: Assume sales data is over the last 5 days
inventory_df["AvgDailySales"] = inventory_df["TotalUnitsSold"] / SALES_PERIOD_DAYS

inventory_df["StockPercentage"] = ((inventory_df["CurrentStock"] / inventory_df["StartingStock"]) * 100).fillna(0)

# Train or load the cached ML model
ml_model = train_depletion_model(inventory_df.copy())

# Predict 'Days to Depletion' using the ML model
if ml_model and not inventory_df.empty:
    features_to_predict = inventory_df[inventory_df['AvgDailySales'] > 0][['CurrentStock', 'AvgDailySales']]
    if not features_to_predict.empty:
        predicted_days = ml_model.predict(features_to_predict)
        inventory_df.loc[inventory_df['AvgDailySales'] > 0, "DaysToDepletion"] = predicted_days.astype(int)
    else:
        inventory_df["DaysToDepletion"] = 999 # Default for non-selling items
else:
    inventory_df["DaysToDepletion"] = 0

# --- Main Inventory Display ---
st.subheader("🗂️ Current Inventory Levels (5-Day Velocity)")
st.info(f"Displaying inventory. 'Est. Days to Depletion' is predicted by an ML model assuming total sales occurred in the last **{SALES_PERIOD_DAYS} days**.")

st.dataframe(
    inventory_df.sort_values("CurrentStock"),
    use_container_width=True,
    column_config={
        "StockPercentage": st.column_config.ProgressColumn(
            "Current Stock Level", help="Percentage of starting stock remaining.", format="%d%%", min_value=0, max_value=100,
        ),
        "Product": st.column_config.TextColumn(width="large"),
        "CurrentStock": "Current Units",
        "DaysToDepletion": st.column_config.NumberColumn(
            "Est. Days to Depletion", help="ML-predicted days until stock runs out based on recent high velocity.", format="%d days",
        ),
        "StartingStock": "Starting Stock",
        # **KEY CHANGE**: Updated column header for clarity
        "TotalUnitsSold": f"Units Sold ({SALES_PERIOD_DAYS}d)",
    },
    column_order=["Product", "StockPercentage", "CurrentStock", "DaysToDepletion", "StartingStock", "TotalUnitsSold"]
)

# --- Reorder Alerts ---
st.subheader("📦 Reorder Alerts")
# Adjusted threshold for a shorter forecast window
REORDER_THRESHOLD_DAYS = 7
low_stock_df = inventory_df[inventory_df["DaysToDepletion"] <= REORDER_THRESHOLD_DAYS]

if not low_stock_df.empty:
    st.error(f"🔥 URGENT: {len(low_stock_df)} products are predicted to deplete in less than {REORDER_THRESHOLD_DAYS} days due to high sales velocity!")
    st.dataframe(
        low_stock_df[["Product", "CurrentStock", "DaysToDepletion"]].sort_values("DaysToDepletion"),
        use_container_width=True
    )
else:
    st.success(f"✅ Inventory levels appear safe for the next {REORDER_THRESHOLD_DAYS} days.")

# --- Alerts and Notifications ---
st.header("⚠️ Business Alerts")

col1, col2 = st.columns(2)

with col1:
    if not prod_df.empty:
        st.subheader("Low Selling Products Alert")
        median_sales = prod_df["UnitsSold"].median()
        threshold = st.slider(
            "Alert threshold (units sold)",
            min_value=0,
            max_value=int(median_sales * 2),
            value=int(median_sales * 0.5)
        )
        
        low_selling = prod_df[prod_df["UnitsSold"] <= threshold]
        if not low_selling.empty:
            st.warning(f"🚨 {len(low_selling)} products with poor sales!")
            st.dataframe(low_selling.sort_values("UnitsSold"), use_container_width=True)
        else:
            st.success("✅ All products performing well!")

with col2:
    if not branch_df.empty:
        st.subheader("Branch Performance Alert")
        
        avg_revenue = branch_df["TotalRevenue"].mean()
        underperforming = branch_df[branch_df["TotalRevenue"] < avg_revenue * 0.7]
        
        if not underperforming.empty:
            st.warning(f"📉 {len(underperforming)} branches underperforming!")
            st.dataframe(underperforming, use_container_width=True)
        else:
            st.success("✅ All branches meeting targets!")

# --- Footer ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Uptime", f"{time.time() - st.session_state.get('start_time', time.time()):.0f}s")

with col2:
    if not st.session_state.get('start_time'):
        st.session_state.start_time = time.time()
    st.metric("Dashboard Runtime", f"{time.time() - st.session_state.start_time:.0f}s")

with col3:
    st.metric("Data Points", f"{len(branch_df) + len(cat_df) + len(prod_df) + len(payment_df)}")

# --- Auto Refresh ---
if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()