import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.stereo_vo import main
    import os
    
    # Ensure results directory exists
    os.makedirs('results/stereo-vo', exist_ok=True)
    
    # Run the comprehensive VO evaluation
    # This executes the main block in main.py which now includes:
    # 1. 01-05 sequences
    # 2. Ablations for 02, 03
    # 3. Trajectory plots and match visualizations
    import subprocess
    import sys
    
    # Use python -m to run correctly from root
    subprocess.run([sys.executable, "-m", "src.stereo_vo.main"])
