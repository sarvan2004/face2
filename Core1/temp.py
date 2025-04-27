
import os
import pandas as pd

# Ensure attendance file exists with proper columns
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_csv(attendance_file, index=False)
    print("✅ Created attendance.csv")

