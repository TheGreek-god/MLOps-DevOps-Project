import os
import pandas as pd
import sys
'''
    Replaces Duplicate Driver/Team names
    Computes rolling average of previous 3 FinishPos for all Drivers.
    All drivers in 2022 Round 1 have an avg of 20.
    All rookie drivers will also be given 20 in their first round.
'''

def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not find project root")

root_path = find_project_root()
src_path = os.path.join(root_path,"src")
if(root_path not in sys.path):
    sys.path.insert(0,root_path)
if(src_path not in sys.path):
    sys.path.insert(0,src_path)

DATAPATH = os.path.join(root_path,"data","Complete_Driver_Data.csv")



def run_data_pipeline(file_path=DATAPATH):
    print("Starting F1Net-V2 Data Preprocessing Pipeline...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing base dataset at {file_path}")


    master_df = pd.read_csv(file_path)

  
    if "Year" not in master_df.columns:
        print("🔄 Index anomaly detected. Resetting 'Year' back to a column...")
        master_df = master_df.reset_index()

    
   
    master_df["Team"] = master_df["Team"].replace("Kick Sauber", "Audi")
    master_df["Driver"] = master_df["Driver"].replace(
        "Andrea Kimi Antonelli", "Kimi Antonelli"
    )

    
    master_df = master_df.sort_values(by=["Driver", "Year", "Round"]).reset_index(
        drop=True
    )

    
    print("Calculating 3-race rolling average for form tracking...")
    master_df["Driver_Form_Avg"] = (
        master_df.groupby("Driver")["FinishPos"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )

    
    master_df["Driver_Form_Avg"] = master_df["Driver_Form_Avg"].fillna(20.0)

    
    print("Restructuring dataset format to race weekend groupings...")
    master_df = master_df.sort_values(by=["Year", "Round", "FinishPos"]).reset_index(
        drop=True
    )

   
    master_df.to_csv(file_path, index=False)
    print(f" Success! Cleaned dataset written back to {file_path}\n")

    
    print("Quick check of 2022 Round 1 Grid Form Baseline:")
    sample = master_df[(master_df["Year"] == 2022) & (master_df["Round"] == 1)]
    print(
        sample[["Year", "Round", "Driver", "Team", "FinishPos", "Driver_Form_Avg"]].head(
            5
        )
    )


if __name__ == "__main__":
    run_data_pipeline()