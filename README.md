Hi Kate and Jack,

Here is my code base for TCFP. It's a bit messy, for which I apologize. I included some of my failed attempts to speed up the processing, on the off-chance Kate would find them useful. I didn't include any of my attempts on Imme's `shannon` machine, as her hardware configuration is different.

Here are the main scripts you might be interested in:
* `test_data.ipynb`: Just playing around with the input TCFP dataset
* `create_simple_model_training_file.ipynb`: Creating a single netCDF file with desired predictors used for training data
* `explore_simple_model_class.ipynb`: A simple random forest classifier model
* `explore_simple_model_class_with_joblib.ipynb`: Same but using `joblib` for better parallelization
* `tcfp_simple_model_tinker.ipynb`: A file where I was playing around with the model parameters of the simple random forest classifier
* `run_simple_model_XGBoost.ipynb`: using XGBoost instead of a random forest for the TCFP model
* 
* `tcfp_basic_eval.ipynb`: Calculating basic stats (e.g., confusion matrix) for standard RF model
* `tcfp_basic_eval_XGB.ipynb`: Calculating basic stats (e.g., confusion matrix) for XGBoost model
* `calc_genesis_incidence_5day.ipynb`: Estimating expected TC genesis probabilities (positive labels) for 1, 5, and 7 day forecasts

I think the preprocessing functions and code in the `create_simple_model_training_file` and `explore_simple_model_*` notebooks will ultimately be the most useful to you. The documentation for the TCFP input predictors is quite thorough--Jack or Chris S. can point you in the right directions. The trained models are in the `tar.gz` file (they will end with `.joblib`). I'm not sure they will work on a different machine--I think they are portable but I can't remember. I figured I'd give them to you anyway.
