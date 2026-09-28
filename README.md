# Indian Rainfall Analysis: Probability and Statistics

An interactive meteorological analytics platform applying classical probability theory, Bayesian inference, continuous distribution fitting, statistical independence testing, and polynomial regression across 115 years of Indian rainfall records (1901-2015).

---

## 1. Problem Statement

India's agricultural economy, drinking water reservoirs, and disaster preparedness depend heavily on the Southwest Monsoon. With more than 1.4 billion people reliant on agrarian output, rainfall uncertainty represents a major socio-economic and climate vulnerability.

Rainfall in India exhibits extreme spatial and temporal stochasticity:
- Spatial Disparity: Annual rainfall ranges from under 100 mm in the arid Thar Desert of Western Rajasthan to over 4,000 mm along the Western Ghats and northeastern hill ranges.
- Temporal Volatility: Year-to-year monsoon intensity fluctuates sharply, creating cycles of droughts and floods.

Intuitive or qualitative assessments cannot accurately capture these meteorological dynamics. Classical probability theory and mathematical statistics provide rigorous tools to move beyond raw observation:
1. Is monsoon rainfall a reliable predictor of total annual rainfall?
2. Are winter precipitation systems (Western Disturbances) and summer monsoon rains statistically independent?
3. Is there a measurable secular trend in annual rainfall over the past century, or is variation dominated by natural noise?

This project implements rigorous statistical models directly in the browser to quantify these questions against official historical data.

---

## 2. Dataset Overview

The study uses two official meteorological datasets sourced from the India Meteorological Department (IMD) via Kaggle.

### Primary Dataset: Subdivisional Historical Rainfall (1901-2015)
- File Name: `rainfall in india 1901-2015.csv`
- Records: 4,116 raw rows; 4,090 have a valid annual total (26 rows contain NA) spanning 115 continuous years
- Spatial Coverage: 36 Meteorological Subdivisions encompassing all states and union territories
- Temporal Range: 1901 through 2015
- Attributes (19 Columns):
  - `SUBDIVISION`: Name of the meteorological subdivision (string)
  - `YEAR`: Calendar year of observation (integer, 1901 to 2015)
  - `JAN` to `DEC`: Monthly cumulative precipitation in millimeters (float)
  - `ANNUAL`: Total yearly rainfall in millimeters (float)
  - `Jan-Feb`: Winter season cumulative rainfall (float)
  - `Mar-May`: Pre-monsoon / summer rainfall (float)
  - `Jun-Sep`: Southwest Monsoon season rainfall (float)
  - `Oct-Dec`: Post-monsoon / retreating monsoon rainfall (float)

### Secondary Dataset: District-Level Normal Rainfall
- File Name: `district wise rainfall normal.csv`
- Records: 641 administrative districts across India
- Usage: Aggregated state-level rainfall normals used to identify the top 10 highest precipitation states

### Why This Dataset Was Selected
1. Statistical Power: 115 continuous years across 36 regions provide large sample sizes (N = 4,090) sufficient for robust distribution fitting and regression without overfitting.
2. Climatological Diversity: Covers all microclimates, from hyper-arid desert zones to tropical rainforests and Himalayan rainbelts.
3. Official Ground Truth: Standardized rain-gauge networks calibrated and maintained by the India Meteorological Department provide dependable observational validity.

---

## 3. Solution and Statistical Methodology

All analytical routines run dynamically in the client browser, parsing raw CSV data on load without relying on precomputed static JSON.

### A. Discrete Random Variable (empirical PMF)
Each annual total is mapped to X in {Deficient, Normal, Excess} using the empirical quartiles (Q1 = 804.5 mm, Q3 = 1644.8 mm, N = 4090):
```
P(X = k) = n_k / N   ->   Deficient 0.250 | Normal 0.500 | Excess 0.250
```
The 25/50/25 split holds by construction of the quartiles, so the PMF illustrates the definition and is not a finding. Computed in `drawPMF()` (analysis.html) via `Stats.pmf3()`.

### B. Continuous Distribution Fitting (MLE)
Jun-Sep rainfall (n = 4106) is fitted with a Normal by maximum likelihood: mu = 1064.7 mm, sigma = 707.7 mm (sigma uses n in the denominator, the MLE). The fit is poor: sample skewness = 1.51 and the Normal assigns 6.6% probability to negative rainfall. A lognormal MLE (ln X ~ N(6.77, 0.64)) is drawn beside it. The histogram shows the chosen subdivision's monsoon rainfall with its own Normal and lognormal MLE curves.

### C. Bayes' Rule
A = {annual > 75th percentile}, B = {Jun-Sep > 75th percentile}, computed over 4090 records with both values:
- P(A) = 0.250, P(B) = 0.250, P(B|A) = 0.910
- Bayes: P(A|B) = P(B|A) P(A) / P(B) = 0.910; direct count = 0.910
- Lift over the 25% baseline: 3.64x. Pooled Jun-Sep share of annual rainfall = 75.3%

The Bayes and direct values agree by algebra (same frequency table), so this is an arithmetic cross-check, not independent validation. The substantive result is the lift.

### D. Independence, Covariance and Correlation (January vs July)
n = 4106 subdivision-year pairs (pooled):
- P(Jan > median) = 0.498, P(Jul > median) = 0.500
- Observed P(both) = 0.267; expected under independence = 0.249; gap = 0.018
- Sample covariance = -466.3 mm^2; Pearson r = -0.0516, r^2 = 0.0027, p = 0.0009 (two-sided t-test, df = n - 2)

With n this large the correlation is statistically significant (p < 0.01) but tiny: January explains about 0.3% of July's variance. The honest conclusion is "practically negligible linear dependence", not proven independence. Zero correlation does not imply independence, and pooling subdivisions with different climates can hide within-region structure.

### E. Polynomial Trend Fitting
Degree 1, 2 or 3 least squares via Gaussian elimination with partial pivoting on the normal equations (A^T A) a = A^T y, with x = year - first year for numerical stability. Rows with a missing year or annual value are dropped together so x and y stay aligned. The page shows R^2, adjusted R^2, and for degree 1 the slope with its t-test p-value.

Kerala (n = 115): linear slope = -2.84 mm/yr (SE 1.16, p = 0.016). R^2 = 0.050 (linear), 0.052 (quadratic), 0.072 (cubic). The decline is statistically significant at the 5% level but explains about 5% of the variance. R^2 always rises with degree, so a higher degree is not evidence of a better model.

### Unit 1 syllabus mapping
| Unit 1 topic | Where |
| :--- | :--- |
| Load / view dataset | dataset.html (Papa.parse, live table) |
| Summary statistics: mean (expectation), variance, quantiles | dataset.html stats bar |
| Discrete random variables, PMF | analysis.html, section A |
| Continuous random variables, densities, MLE | analysis.html, section B |
| Bayes' rule, conditional probability | analysis.html, section C |
| Independence, covariance | analysis.html, section D |
| Curve fitting (polynomial) | analysis.html, section E |

Machine-learning framing: the trend model is supervised learning (regression of annual rainfall on year). The descriptive statistics and distribution fits are exploratory analysis. Not covered by the code: the theory-only topics (conditional independence, joint continuous distributions).

---

## 4. Tech Stack and Architecture

The platform is built as a zero-dependency, ultra-lightweight client-side application designed for instant loading and high portability.

- Core Markup: Semantic HTML5 with accessibility attributes and SEO metadata
- Styling: Custom Vanilla CSS3 implementing a bespoke palette (forest green `#013e37` and cream `#ffefb3`), responsive glassmorphism, flexbox/grid layouts, and zero heavy CSS frameworks
- Logic: Modern Vanilla JavaScript (ES6+) for math routines, matrix inversion, and quantile estimations
- Data Processing: PapaParse (v5.4.1) for client-side streaming and parsing of multi-megabyte CSV files directly in memory
- Visualization: Chart.js (v4.4.2) for responsive line trends, scatter plots, histograms, and horizontal bar charts
- Interactive Canvas: Custom HTML5 Canvas rendering real-time particle rain simulations and meteorological isobar contours
- Hosting Target: Vercel (Zero-configuration static site deployment)

---

## 5. Project Structure

```
PRO_1_live/
|-- .gitignore                             # Git ignore rules for clean deployment
|-- README.md                              # Complete project documentation
|-- vercel.json                            # Vercel static deployment configuration
|-- index.html                             # Landing page: problem statement, overview, visual plates
|-- dataset.html                           # Dataset schema, summary statistics bar, live data table
|-- analysis.html                          # Interactive statistical workbench: Bayes, independence, trends
|-- stats.js                               # Shared math: quantiles, MLE, Bayes, Pearson, regression tests
|-- test_stats.js                          # Node script that prints every number quoted here
|-- style.css                              # Design system tokens, layouts, and responsive styles
|-- rainfall_bg.jpg                        # Meteorological backdrop image for problem statement section
|-- rainfall in india 1901-2015.csv        # Primary IMD subdivisional dataset (115 years)
`-- district wise rainfall normal.csv      # Secondary IMD district normal dataset (641 districts)
```

---

## 6. How to Run Locally

Because the application fetches CSV datasets dynamically via HTTP requests using PapaParse, it must be viewed through a local web server (not via `file://`).

### Option A: Using Python (Recommended)
Open a terminal in the project directory and run:
```bash
python -m http.server 8000
```
Then visit: `http://localhost:8000`

### Option B: Using Node.js / npx
```bash
npx serve .
```

### Option C: Using VS Code
Install the **Live Server** extension, right click `index.html`, and select **Open with Live Server**.

---

## 7. Hosting on Vercel

The project is structured for native, zero-configuration deployment on Vercel.

Push the folder to GitHub and import it in Vercel (Framework Preset: Other, no build command), or run `npx vercel` in this folder.


---

## 8. Summary of Key Analytical Findings

| Dimension | Method | Result | Interpretation |
| :--- | :--- | :--- | :--- |
| Predictive value of monsoon | P(High Annual \| High Monsoon) | 0.910 (3.64x baseline) | Top-quartile monsoon strongly signals top-quartile annual rainfall |
| Monsoon share | sum(Jun-Sep) / sum(Annual) | 75.3% | Monsoon supplies about three quarters of pooled rainfall |
| Jan vs Jul dependence | Pearson r (pooled) | r = -0.052, p = 0.0009 | Significant but negligible (r^2 = 0.0027) |
| Median-split gap | \|P(both) - P(Jan)P(Jul)\| | 0.018 | Small departure from independence |
| Kerala trend | Linear least squares | -2.84 mm/yr, p = 0.016, R^2 = 0.050 | Significant decline, weak explanatory power |
| Monsoon distribution | Normal MLE vs lognormal | skew 1.51 | Right-skewed; Normal fit poor |
| Annual distribution | Mean vs median | 1411 vs 1121 mm | Right-skewed |

## 9. Reproducibility and Limitations
- Every number above is produced by `stats.js`. Run `node test_stats.js` in this folder to reprint them; the web pages call the same functions.
- Observations are pooled across subdivisions with very different climates and are not independent draws (neighbouring subdivisions and consecutive years are correlated), so p-values are optimistic.
- Rainfall is not normally distributed; Normal-based summaries should be read with the skewness caveat.
- Quartile thresholds are computed on the pooled sample, not per subdivision.
