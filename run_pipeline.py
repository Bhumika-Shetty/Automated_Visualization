from dotenv import load_dotenv
load_dotenv()

from pipeline_v3 import Pipeline

if __name__ == "__main__":
    # Full run: 30 runs per combo
    p = Pipeline(num_runs=30)
    p.run()
