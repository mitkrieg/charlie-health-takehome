# Summary

## Problem

Assign ~2,172 mental health patients into treatment cohorts of ~12 for Charlie Health's Intensive Outpatient Program (IOP). Groups are formed once on an initial population and must then accept new patients on a rolling basis without rebuilding from scratch.

---

## Data

2,172 patients split **85 / 15** into an initial set (1,847) and a newcomer holdout (325) to simulate rolling enrollment. Feature vectors are built by `Vectorizer`: 19 scaled numerics, 4 one-hot categoricals, and composite engineered features (`stress_index`, `wellbeing_score`, `worklife_balance`, `high_risk`). Patients where `treatment_not_needed = 1` (low depression, low pressure, high satisfaction, no suicidal ideation, adequate sleep and diet) are routed to a reserved group −1 and excluded from all clustering metrics.

---

## EDA Findings

- **Population:** Roughly equal split of students and working professionals, ages 18–60 nearly uniformly distributed.
- **Depression prevalence:** 17.8% of patients carry the depression flag — a minority but the primary clinical signal.
- **Depression affects gender equally.** Depression is similarly distributed among depressed and non-depressed patients, younger people (and by proxy students) have much higher rates of depression.
- **Clinical signals:** When depression is present, pressure and financial stress are markedly higher and satisfaction is lower. These features separate the two populations much more cleanly than demographic features alone.
- **Pressure and satisfaction are inversely related** — a natural axis for group separation.
- **Students report longer work/study hours** than professionals on average. Financial stress peaks for professionals in the 23–30 age band.
- **Education level:** High-school-only patients are over-represented in depression, likely a proxy for age and fewer coping resources rather than education itself.
- **Profession category:** Stress index is relatively uniform across professions — the *type* of work matters less than the *intensity*.

---

## Modeling

### Approach

Agglomerative clustering is a natural fit for this task. The goal is to find locally similar patients and merge them bottom-up, without requiring a pre-specified centroid count. This is preferable to k-means for small, heterogeneous cohorts where the number of natural groups is not known in advance.

Ward linkage (minimises within-cluster variance) is used over average linkage. Average linkage produces better global pairwise similarity but creates singleton and very small clusters, which is a worse patient experience — being an outlier in a group matters clinically.

### Experiments

| Model | Groups | Mean WCSS | Silhouette | Notes |
|---|---|---|---|---|
| **Baseline (random)** | 181 | ~2,039 | -0.35  | No cohesion by design; confirms the need for structure |
| **Agglomerative (ward)** | 264 | ~51 | 0.06 | 40× WCSS reduction; group count rises due to cap enforcement splits |
| **Agglomerative Average linkage** | 308 | 0.07 | more negative | Better global WCSS but produces singletons |
| **Clinical weights (ward)** | 274 | ~50 | 0.05 | Upweighting depression / suicidal thoughts / stress pulls clinically similar patients together but slightly hurts demographic uniformity |
| **k-NN connectivity** | 265 | ~50 | 0.06 | Local neighbor constraint reduces bridging artifacts; modest gain |
| **Demographic connectivity** (`is_professional × age_group × gender_enc`) | 278 |~55 | 0.01 | Hard structural constraint significantly boosts demographic cohesion; WCSS rises slightly as the trade-off |
| **Clinical weights + demographic connectivity** | 273 | ~54 | 0.02 | Best balance: clinically weighted distances enforce symptom similarity; demographic connectivity ensures merge eligibility is demographically sensible |

### Why connectivity constraints?

Without a connectivity matrix, ward linkage can bridge demographically or clinically dissimilar patients through a chain of intermediate points. Constraining merge eligibility to patients who share certain attributes enforces a certain level of matching, rather than relying on the distance metric alone to separate groups that differ on these dimensions.

---

## Newcomer Assignment

Newcomers are assigned online one at a time to the nearest cluster (by weighted Euclidean distance to centroid) with remaining capacity. When all clusters are full, the nearest cluster is split using agglomerative clustering.

Mean WCSS grows steadily with each newcomer added. At 325 newcomers this growth is modest. Two risks scale with the newcomer pool:

1. **Capacity pressure** — as more clusters hit the 12-patient cap, splits become more frequent, increasing the WCSS cost per new arrival.
2. **Centroid drift** — high-traffic clusters accumulate the most drift; periodic re-clustering of the full population would be needed at scale.

---

## Deployment

The final model is served via a FastAPI microservice in `./api`. See `examples/api_workflow.py` for an end-to-end usage example.

**Key endpoints:** `POST /initialize` (fit model on CSV upload), `GET /groups`, `POST /newcomers`, `GET /metrics`, patient CRUD, group split/delete, similar-group lookup.

### Road to Production

- Replace in-memory state with a persistent store (database or object storage) for groups and patient assignments
- Add model versioning so re-trains can be staged and rolled back
- Extend the model to handle patient departures and program completions
- Solicit clinical input on weight tuning and any hard constraints (e.g. should two high-risk patients ever share a group? Does severity mixing help or hurt outcomes?)
- Add a human feedback loop: allow clinicians to override or annotate assignments, and feed those signals back into weight calibration
- Monitor cluster health over time (WCSS, cohesion, override rate) and define thresholds that trigger re-alignment
- Evaluate against clinical outcomes (program completion, session attendance, escalations) once longitudinal data is available

### Open Questions

- Does higher within-group similarity always lead to better outcomes, or is some diversity of experience beneficial?
- How should the model handle changes in patient population mix over time (e.g. more professionals, new program types)?
- Are there fairness concerns — are men and women, students and professionals receiving equivalent grouping quality?
