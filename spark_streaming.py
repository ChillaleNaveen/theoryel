import os
from datetime import datetime
from typing import Optional
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession

# --- Stateful Aggregation Functions ---
def update_revenue(new_values, running_total):
    return sum(new_values, running_total or 0)

# --- Save Function ---
def save_rdd(time, rdd, path, schema):
    count = rdd.count()
    print(f"[INFO] RDD at {time}, count = {count}")
    if count == 0:
        print(f"[INFO] Skipping empty RDD for {path}")
        return

    try:
        spark = SparkSession.builder.getOrCreate()

        def to_row(record):
            key, value = record
            if isinstance(key, tuple):
                # Create row dict with proper field mapping
                row_dict = {}
                for i, field_name in enumerate(schema[:-1]):  # All fields except the last one
                    row_dict[field_name] = key[i]
                row_dict[schema[-1]] = value  # Last field is the aggregated value
            else:
                # Single key case
                row_dict = {schema[0]: key, schema[1]: value}
            return Row(**row_dict)

        row_rdd = rdd.map(to_row)
        df = spark.createDataFrame(row_rdd)  # Remove schema param, let Spark infer
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
        print(f"[SAVED] Data written to: {path} at {time}")
    except Exception as e:
        print(f"[ERROR] Failed to write {path} at {time}: {e}")

def save_dstream(dstream, path, schema):
    dstream.foreachRDD(lambda time, rdd: save_rdd(time, rdd, path, schema))

# --- Line Parser ---
def parse_line(line: str) -> Optional[Row]:
    print(f"[RECEIVED] {line.strip()}")
    fields = line.strip().split(",")
    if len(fields) != 16:
        print(f"[SKIPPED] Wrong number of columns: expected 16, got {len(fields)}")
        return None
    try:
        # More robust timestamp parsing
        timestamp_str = fields[1].strip()
        try:
            ts = datetime.strptime(timestamp_str, "%d-%m-%Y %H:%M")
        except ValueError:
            # Try alternative format
            ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        
        hour = ts.hour
        week = ts.strftime('%Y-W%U')
        month = ts.strftime('%Y-%m')
        time_slot = (
            "Morning" if 6 <= hour < 12 else
            "Afternoon" if 12 <= hour < 17 else
            "Evening" if 17 <= hour < 21 else
            "Night"
        )
        
        # More robust numeric parsing
        quantity = int(float(fields[9].strip()))
        final_amount = float(fields[13].strip())
        
        return Row(
            BranchName=fields[3].strip(),
            Category=fields[7].strip(),
            Product=fields[8].strip(),
            Quantity=quantity,
            FinalAmount=final_amount,
            PaymentType=fields[15].strip(),
            TimeSlot=time_slot,
            Week=week,
            Month=month
        )
    except Exception as e:
        print(f"[ERROR] Parsing failed for line: {line.strip()[:100]}... Error: {e}")
        return None

# --- Main Streaming Logic ---
def main():
    output_dirs = [
        "output/branch_sales", "output/category_sales", "output/product_sales",
        "output/payment_type_analysis", "output/branch_time_demand",
        "output/weekly_branch_revenue", "output/monthly_branch_revenue"
    ]
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    sc = SparkContext("local[2]", "LiveInsightRetail")
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 2.0)  # Increased batch interval for stability
    ssc.checkpoint("checkpoint_dir")

    lines = ssc.socketTextStream("localhost", 9999)
    parsed = lines.map(parse_line).filter(lambda x: x is not None)

    # Core Aggregations
    branch_sales = parsed.map(lambda r: (r.BranchName, r.FinalAmount)).updateStateByKey(update_revenue)
    category_sales = parsed.map(lambda r: (r.Category, r.FinalAmount)).updateStateByKey(update_revenue)
    product_sales = parsed.map(lambda r: (r.Product, r.Quantity)).updateStateByKey(update_revenue)
    payment_counts = parsed.map(lambda r: (r.PaymentType, 1)).updateStateByKey(update_revenue)
    branch_time = parsed.map(lambda r: ((r.BranchName, r.TimeSlot), r.FinalAmount)).updateStateByKey(update_revenue)
    product_inventory = parsed.map(lambda r: (r.Product, r.Quantity)).updateStateByKey(update_revenue)
    save_dstream(product_inventory, "output/product_inventory_usage", ["Product", "TotalUnitsSold"])


    # Weekly and Monthly Revenue per Branch
    weekly_branch = parsed.map(lambda r: ((r.BranchName, r.Week), r.FinalAmount)).updateStateByKey(update_revenue)
    monthly_branch = parsed.map(lambda r: ((r.BranchName, r.Month), r.FinalAmount)).updateStateByKey(update_revenue)

    # Save all outputs
    save_dstream(branch_sales, "output/branch_sales", ["BranchName", "TotalRevenue"])
    save_dstream(category_sales, "output/category_sales", ["Category", "TotalRevenue"])
    save_dstream(product_sales, "output/product_sales", ["Product", "UnitsSold"])
    save_dstream(payment_counts, "output/payment_type_analysis", ["PaymentType", "TransactionCount"])
    save_dstream(branch_time, "output/branch_time_demand", ["BranchName", "TimeSlot", "TotalRevenue"])
    save_dstream(weekly_branch, "output/weekly_branch_revenue", ["BranchName", "Week", "TotalRevenue"])
    save_dstream(monthly_branch, "output/monthly_branch_revenue", ["BranchName", "Month", "TotalRevenue"])

    ssc.start()
    print("[INFO] Streaming context started. Waiting for data...")
    ssc.awaitTermination()

if __name__ == "__main__":
    main()