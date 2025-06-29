# File: app.py
# This file is part of the Face Recognition Based Attendance System project.

import gradio as gr
import pandas as pd
import os
from datetime import datetime



# Directory where attendance CSVs are stored
ATTENDANCE_DIR = "Attendance"

# Get list of available attendance dates
def get_dates():
    files = os.listdir(ATTENDANCE_DIR)
    dates = []
    for f in files:
        if f.startswith("Attendance_") and f.endswith(".csv"):
            date_part = f[len("Attendance_"):-len(".csv")]
            dates.append(date_part)
    dates.sort(reverse=True)
    return dates

# Load CSV based on selected date and filter
def load_attendance(selected_date, search=""):
    file_path = os.path.join(ATTENDANCE_DIR, f"Attendance_{selected_date}.csv")
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["NAME", "USER_ID", "TIME", "DATE", "STATUS"])
    
    df = pd.read_csv(file_path)
    if search:
        search_lower = search.lower()
        df = df[df.apply(lambda row: search_lower in str(row['NAME']).lower() or search_lower in str(row['USER_ID']).lower(), axis=1)]
    return df

# Interface components
with gr.Blocks() as demo:
    gr.Markdown("## 📊 Face Recognition Attendance Dashboard")
    
    with gr.Row():
        selected_date = gr.Dropdown(choices=get_dates(), label="Select Date", value=datetime.now().strftime("%d-%m-%Y"))
        search_box = gr.Textbox(label="🔍 Search by Name or ID", placeholder="e.g. Abhay or 1")

    table = gr.Dataframe(headers=["NAME", "USER_ID", "TIME", "DATE", "STATUS"], interactive=False)

    def refresh_dashboard(date, search):
        return load_attendance(date, search)

    selected_date.change(fn=refresh_dashboard, inputs=[selected_date, search_box], outputs=table)
    search_box.change(fn=refresh_dashboard, inputs=[selected_date, search_box], outputs=table)

    gr.Button("🔄 Refresh").click(fn=refresh_dashboard, inputs=[selected_date, search_box], outputs=table)

# Run the app
# demo.launch()
demo.launch(share=True)
