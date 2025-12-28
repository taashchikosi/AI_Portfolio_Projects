Model Architecture & Performance
1) Model Structure
Two independent models:
	•	rf_occ: predicts next-24h maximum occupancy ratio 
	•	rf_wait: predicts next-24h mean wait time 
Model type:
	•	RandomForestRegressor 
	•	Deployment-optimised (80 trees, capped depth) 

2) Target Variables
	•	target_max_occ_ratio_24h 
	•	target_mean_wait_24h 

3) Evaluation Metrics (Deployment Model)
Next-24h Max Occupancy Ratio
	•	MAE: 0.012 
	•	RMSE: 0.017 
Next-24h Mean Wait Minutes
	•	MAE: 4.78 minutes 
	•	RMSE: 6.11 minutes 
These metrics are appropriate for screening-level forecasting, not precision scheduling.

4) When Performance May Degrade
	•	Sudden demand shocks 
	•	Policy changes not represented in training data 
	•	Structural staffing changes 
	•	Seasonal regime shifts 

5) Improvement Roadmap
	•	Integrate discharge pipeline signals 
	•	Add arrival source decomposition 
	•	Introduce probabilistic uncertainty bands 
