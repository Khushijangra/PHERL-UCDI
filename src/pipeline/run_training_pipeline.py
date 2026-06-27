import os
import subprocess
import sys

def get_base_dir():
    # src/pipeline/run_training_pipeline.py -> src/pipeline -> src -> project_root
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def run_script(script_name):
    script_path = os.path.join(get_base_dir(), 'src', 'models', script_name)
    print(f"\n{'='*50}\nExecuting: {script_name}\n{'='*50}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ SUCCESS: {script_name}")

def orchestrate_training():
    print("🚀 PHERL-UCDI++ Lightning GPU Pipeline Entry Point")
    
    # Verify dataset exists
    data_path = os.path.join(get_base_dir(), 'data', 'processed', 'training_dataset.pt')
    if not os.path.exists(data_path):
        print(f"❌ CRITICAL ERROR: {data_path} not found.")
        print("Please upload the packaged dataset to Lightning AI before running.")
        sys.exit(1)
        
    run_script('train_baselines.py')
    run_script('train_gnn.py')
    run_script('compare_models.py')
    
    print("\n🎉 ML Training Pipeline Completed!")
    print("Artifacts generated in /models/ and /reports/.")
    print("Please download best_model.pt and CSVs back to the local development node.")

if __name__ == "__main__":
    orchestrate_training()
