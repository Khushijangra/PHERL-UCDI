import os
import subprocess
import sys

def get_base_dir():
    # src/pipeline/run_training_pipeline.py -> src/pipeline -> src -> project_root
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def run_script(script_name):
    module_name = f"src.models.{script_name[:-3]}"
    print(f"\n{'='*50}\nExecuting Module: {module_name}\n{'='*50}")
    
    result = subprocess.run([sys.executable, "-m", module_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {module_name} exited with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ SUCCESS: {module_name}")

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
    
    # Run validation scripts
    print(f"\n{'='*50}\nExecuting Module: src.validation.sanity_verifier\n{'='*50}")
    result = subprocess.run([sys.executable, "-m", "src.validation.sanity_verifier"], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: src.validation.sanity_verifier exited with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ SUCCESS: src.validation.sanity_verifier")
        
    print(f"\n{'='*50}\nExecuting Module: src.validation.explainability\n{'='*50}")
    result = subprocess.run([sys.executable, "-m", "src.validation.explainability"], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: src.validation.explainability exited with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ SUCCESS: src.validation.explainability")
        
    print(f"\n{'='*50}\nExecuting Module: src.validation.counterfactual\n{'='*50}")
    result = subprocess.run([sys.executable, "-m", "src.validation.counterfactual"], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: src.validation.counterfactual exited with code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ SUCCESS: src.validation.counterfactual")
    
    print("\n🎉 ML Training Pipeline Completed!")
    print("Artifacts generated in /models/ and /reports/.")
    print("Please download best_model.pt and CSVs back to the local development node.")

if __name__ == "__main__":
    orchestrate_training()
