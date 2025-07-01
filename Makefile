.PHONY: env train eval figs demo package paper

# Python virtual environment setup
env:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# Train adaptive-k predictor and centroid refinement models
train:
	. .venv/bin/activate && python src/adaptive_k.py --train && python src/refine_centroid.py --train

# Run baselines and our method on test images
eval:
	. .venv/bin/activate && python scripts/run_evaluation.py

# Generate result figures
figs:
	. .venv/bin/activate && python scripts/plot_results.py

# Launch Streamlit demo (headless; Ctrl-C to stop)
demo:
	. .venv/bin/activate && streamlit run demo/app.py

# Create submission zip without bulky image outputs or venv
package:
	zip -r submission_package.zip src scripts models results/metrics.csv figures manuscript demo README.md requirements.txt Makefile -x "results/compressed/*" ".venv/*"

# Build manuscript PDF
paper:
	tectonic manuscript/main.tex 