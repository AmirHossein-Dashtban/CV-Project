import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.stereo import main
    frames = main.get_all_frames()
    main.save_visualizations(frames)
    summary = main.ablation_study(frames)
    failures = main.find_failures(frames)
    
    print("\nProcessing complete. Updates for report:")
    print("Ablation Table Data:")
    for c in summary:
        for w in summary[c]:
            bpr, mae = summary[c][w]
            print(f"{c}, {w}: {bpr:.2f}, {mae:.2f}")
            
    print("\nFailure Table Data:")
    for f, bpr, mae in failures:
        print(f"{f[0]}_{f[1]}.png: {bpr:.2f}, {mae:.2f}")
