import fastf1
from fastf1.events import get_event_schedule
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not find project root")

root_path = find_project_root()

cache_folder = os.path.join(root_path, 'cache_folder')
os.makedirs(cache_folder, exist_ok=True)

fastf1.Cache.enable_cache(cache_folder)

data = []

year = 2026 #change this manually to the required year-> This is done to overcome api call limits
time_cols = ['AvgFPTime','QualyTime', 'QualTimeDelta']

print(f"Processing year: {year}")
try:
    schedule = get_event_schedule(year)
except Exception as e:
    print(f"❌ Failed to load schedule for {year}: {e}")


for _,event in tqdm(schedule.iterrows(),total = len(schedule),desc=f"{year} Season"):
    round_num = event['RoundNumber']
    event_name = event['EventName']
    print(f'Round {round_num} : {event_name}')

    try:

        
        #get all session details
        sessionR = fastf1.get_session(year,round_num,'R')
        sessionR.load()
        time.sleep(5)
        sessionQ = fastf1.get_session(year, round_num, 'Q')
        sessionQ.load()
        quali_results = sessionQ.results.set_index("DriverNumber")
        time.sleep(5)

        all_fp_laps = []

        # Load FP1 (required)
        sessionFP1 = fastf1.get_session(year, round_num, 'FP1')
        sessionFP1.load()
        time.sleep(5)
        all_fp_laps.append(sessionFP1.laps)

        # Try FP2 and FP3
        for fp_type in ['FP2', 'FP3']:
            try:
                session = fastf1.get_session(year, round_num, fp_type)
                session.load()
                time.sleep(5)
                all_fp_laps.append(session.laps)
            except ValueError:
                pass


        all_fp_laps = pd.concat(all_fp_laps, ignore_index=True)


        #get average Free Practice Times
        
        avg_fp = all_fp_laps.groupby('Driver')["LapTime"].mean().dt.total_seconds()

        print("\nFP data Loaded\n")
        
        #Get Qualy deltas

        q_laps_fastest = sessionQ.laps.pick_fastest()
        q_laps = sessionQ.laps.pick_not_deleted()

        if isinstance(q_laps,pd.Series):
            q_laps = q_laps.to_frame().T

        fastest_time = q_laps_fastest["LapTime"]
        
        valid_laps = q_laps[q_laps["LapTime"].notna()]

        fastest_laps = valid_laps.loc[valid_laps.groupby('Driver')['LapTime'].idxmin()]

        fastest_laps["LapDelta"] = fastest_laps["LapTime"].apply(lambda x: x-fastest_time).dt.total_seconds()

        fastest_laps["LapTime"] = fastest_laps["LapTime"].dt.total_seconds()

        #qualyData = pd.DataFrame(fastest_laps[["Driver","LapTime","LapDelta","IsAccurate"]])


        
        #RaceResults

        results = sessionR.results.set_index("DriverNumber")


        #Create Rows
        for drv in results.index:
            row = results.loc[drv]
            driver_code = row["Abbreviation"]
            driver_name = row["FullName"]
            team_name = row["TeamName"]
            finishing_pos = row["Position"]
            q_position = quali_results.loc[drv, "Position"] if drv in quali_results.index else None
            grid_pos = row["GridPosition"] if (pd.notna(row["GridPosition"]) and row["GridPosition"] != 0 and row["GridPosition"] != -1) else q_position
            
            fp_time = avg_fp.get(driver_code)
            q_data = fastest_laps[fastest_laps["Driver"] == driver_code]

            if not q_data.empty:
                qualyDelta = q_data.iloc[0]["LapDelta"]
                qualyTime = q_data.iloc[0]["LapTime"]
                is_accurate = q_data.iloc[0]["IsAccurate"]
            else:
                qualyDelta = None
                qualyTime = None
                is_accurate = None


            data.append({
                "Year":year,
                "Round":round_num,
                "Driver":driver_name,
                "Team":team_name,
                "AvgFPTime":fp_time,
                "QualyTime":qualyTime,
                "QualTimeDelta":qualyDelta,
                "GridPos":grid_pos,
                "FinishPos":finishing_pos,
                "IsAccurate":is_accurate
            } )

            print("Row Appended")
        

    except Exception as e:
        print(f" ⚠️ Skipping {year} Round {round_num} ({event_name}): {e}")

print(f"{year} Done")





#Convert to pandas DF
df = pd.DataFrame(data)
df.to_csv(os.path.join(root_path, "data", f"Driver_race_data_{year}.csv"), index=False)
print(f"✅ Data collection complete. Saved to 'driver_race_data_{year}.csv'")