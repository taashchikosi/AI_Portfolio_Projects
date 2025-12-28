Data & Feature Design
1) Feature Schema (Inputs to the Model)
Capacity & Load
	•	capacity_beds: Total staffed bed capacity 
	•	occupancy_ratio: Current occupied beds ÷ capacity 
	•	arrivals: Estimated hourly arrivals 
	•	wait_minutes: Current mean patient wait time 
Operational Modifiers
	•	staff_index: Relative staffing availability (scaled) 
Time Context
	•	hour, dayofweek, month, is_weekend 
Rolling 24-Hour Context
	•	arrivals_roll_mean_24h, arrivals_roll_max_24h 
	•	occ_roll_mean_24h, occ_roll_max_24h 
	•	wait_roll_mean_24h, wait_roll_max_24h 

2) Why Rolling 24-Hour Features Exist
Short-term congestion is path-dependent. Rolling features capture:
	•	momentum 
	•	sustained pressure 
	•	recent spikes 
In the demo, these are conservatively approximated to avoid overstating certainty.

3) Train/Test Split Logic
	•	Chronological split at the 80th percentile timestamp 
	•	Prevents look-ahead bias 
	•	Mimics real-world forecasting conditions 

4) Data Limitations
	•	No patient-level acuity 
	•	No real-time discharge pipeline 
	•	No external shocks (mass casualty events) 
These limitations are intentional for a screening-level tool.
