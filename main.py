# main.py
import subprocess
import sys

def run_step(script_path, phase_name):
    print(f"\n{'='*60}")
    print(f"Executing {phase_name}: {script_path}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline halted. {script_path} failed.")
        sys.exit(1)

if __name__ == "__main__":
    print("Initializing Machine Learning Pipeline...")

    # Phase 1: Theory is handled in the EDA notebook, no execution needed here.
    
    run_step("02_Math/data_prep.py", "PHASE 2 - MATH (Data Cleaning)")
    run_step("02_Math/build_features.py", "PHASE 2 - MATH (Feature Encoding)")
    
    run_step("03_Implementation/train_tuned.py", "PHASE 3 - IMPLEMENTATION (Model Training)")
    
    run_step("04_Visualization/explain.py", "PHASE 4 - VISUALIZATION (Interpretability)")
    
    print("\nPipeline execution completed successfully.")