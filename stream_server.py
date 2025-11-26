# File: stream_server.py
import socket
import time
import pandas as pd

def send_data(file_path, host='localhost', port=9999, delay=0.1):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.listen(1)
        print(f"[SERVER] Listening on {host}:{port}...")
        conn, addr = s.accept()
        print(f"[SERVER] Connected to {addr}")

        df = pd.read_csv(file_path)

        print("[SERVER] Streaming data...")
        for _, row in df.iterrows():
            row = row.copy()

            # Fix timestamp format
            row["Timestamp"] = pd.to_datetime(row["Timestamp"]).strftime("%d-%m-%Y %H:%M")

            # Ensure integers are sent cleanly (not 4.0)
            row["Quantity"] = int(float(row["Quantity"]))
            row["BranchID"] = int(float(row["BranchID"]))
            row["TotalBranches"] = int(float(row["TotalBranches"]))

            line = ",".join(map(str, row.values)) + "\n"
            conn.sendall(line.encode("utf-8"))
            print(f"Sent: {line.strip()}")
            time.sleep(delay)

        print("[SERVER] Finished sending all data.")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        s.close()
        print("[SERVER] Socket closed.")

if __name__ == "__main__":
    send_data("retail_data_bangalore.csv")
