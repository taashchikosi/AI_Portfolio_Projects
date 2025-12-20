# Data Overview — Energy Efficiency Decision-Support Model

## 1. Purpose
This document explains the nature of the data used to develop and test the energy-efficiency decision-support model.

The objective is to:
- Provide transparency about data origins
- Clarify how the data should be interpreted
- Set appropriate expectations regarding model reliability

This data is used to support **directional insights**, not certified outcomes.

---

## 2. Data Source Type
The dataset used in this project is **synthetic and simulation-informed**, designed to reflect realistic patterns observed in commercial buildings.

Synthetic data was selected to:
- Avoid confidentiality and licensing constraints
- Enable controlled experimentation across scenarios
- Ensure reproducibility for demonstration and learning purposes

The structure and distributions are informed by:
- Common building energy performance relationships
- Industry-standard energy-efficiency principles
- Typical ranges observed in real-world building portfolios

---

## 3. Dataset Scope
The dataset represents a portfolio of buildings with varying characteristics, including:
- Building size and occupancy type
- Baseline energy intensity
- HVAC efficiency levels
- Lighting system performance
- Operational controls maturity

Each record corresponds to a hypothetical building or scenario configuration.

---

## 4. Data Generation Logic
Synthetic data was generated to preserve realistic relationships, such as:
- Higher baseline energy use correlating with greater savings potential
- HVAC efficiency having a stronger influence than minor envelope changes
- Operational controls influencing realised performance outcomes
- Diminishing returns at higher efficiency levels

Random variation was introduced to reflect natural heterogeneity across buildings.

---

## 5. Data Quality Considerations
While synthetic, the data is designed to be:
- Internally consistent
- Free from missing values
- Constrained to plausible physical and operational ranges

However, it does **not** capture:
- Installation quality variability
- Behavioural changes post-intervention
- Short-term operational anomalies
- Extreme or atypical building archetypes

---

## 6. Limitations
Key limitations of the dataset include:
- Lack of direct linkage to measured, metered building data
- Simplification of complex physical interactions
- Absence of temporal dynamics (e.g. weather variability over time)

As a result, outputs should be interpreted as **indicative**, not predictive guarantees.

---

## 7. Governance and Responsible Use
This dataset is suitable for:
- Model development and testing
- Scenario exploration
- Demonstration of AI decision-support workflows

It is **not suitable** for:
- Regulatory reporting
- Certified energy assessments
- Formal ESG disclosures without additional validation

Human judgment remains essential when interpreting results.

---

## 8. Future Data Enhancements
Potential future improvements include:
- Integration of open benchmarking datasets
- Incorporation of measured operational data
- Enhanced temporal and climate sensitivity
- Calibration against real-world case studies

These enhancements would further improve realism and confidence.
