/* Shared, DOM-free statistics helpers. Loaded by the pages and by the Node test (test_stats.js). */
(function (root) {
  const isNum = v => typeof v === 'number' && isFinite(v);
  const sum = a => a.reduce((s, v) => s + v, 0);
  const mean = a => sum(a) / a.length;
  const sampleVar = a => { const m = mean(a); return sum(a.map(v => (v - m) ** 2)) / (a.length - 1); };

  // Linear-interpolation quantile on a sorted array (same as pandas/numpy default).
  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q, base = Math.floor(pos), rest = pos - base;
    return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
  }
  const sortedCopy = a => [...a].sort((x, y) => x - y);

  function skewness(a) {
    const m = mean(a), s = Math.sqrt(sum(a.map(v => (v - m) ** 2)) / a.length);
    return mean(a.map(v => ((v - m) / s) ** 3));
  }
  // MLE for Normal: mu = sample mean, sigma^2 = (1/n) * sum (x-mu)^2  (n, not n-1)
  function normalMLE(a) { const mu = mean(a); return { mu, sigma: Math.sqrt(sum(a.map(v => (v - mu) ** 2)) / a.length) }; }
  // MLE for lognormal: Normal MLE of ln(x) (positive values only)
  function lognormalMLE(a) { const l = a.filter(v => v > 0).map(Math.log); return normalMLE(l); }
  const normPdf = (x, mu, s) => Math.exp(-((x - mu) ** 2) / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));
  const lognormPdf = (x, mu, s) => x > 0 ? Math.exp(-((Math.log(x) - mu) ** 2) / (2 * s * s)) / (x * s * Math.sqrt(2 * Math.PI)) : 0;
  function erf(x) { // Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7)
    const t = 1 / (1 + 0.3275911 * Math.abs(x));
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return x >= 0 ? y : -y;
  }
  const normCdf = (x, mu, s) => 0.5 * (1 + erf((x - mu) / (s * Math.SQRT2)));

  // Regularised incomplete beta -> exact two-sided t-test p-values (Numerical Recipes betacf).
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015;
    for (let j = 0; j < 6; j++) s += c[j] / ++y;
    return -t + Math.log(2.5066282746310005 * s / x);
  }
  function betacf(a, b, x) {
    const FP = 1e-300; const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-12) break;
    }
    return h;
  }
  function incBeta(x, a, b) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  const tTwoSidedP = (t, df) => incBeta(df / (df + t * t), df / 2, 0.5);

  // Pearson r, sample covariance (n-1), and p-value for H0: rho = 0.
  function pearson(xs, ys) {
    const n = xs.length, mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
    const r = sxy / Math.sqrt(sxx * syy);
    return { n, r, r2: r * r, cov: sxy / (n - 1), p: tTwoSidedP(r * Math.sqrt((n - 2) / (1 - r * r)), n - 2) };
  }
  // Simple OLS with slope t-test.
  function linreg(xs, ys) {
    const n = xs.length, mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0;
    for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
    const slope = sxy / sxx, intercept = my - slope * mx;
    const sse = sum(xs.map((x, i) => (ys[i] - intercept - slope * x) ** 2));
    const se = Math.sqrt(sse / (n - 2) / sxx);
    return { slope, intercept, se, p: tTwoSidedP(slope / se, n - 2) };
  }
  const adjR2 = (r2, n, deg) => 1 - (1 - r2) * (n - 1) / (n - deg - 1);
  const fmtP = p => p < 0.001 ? p.toExponential(1) : p.toFixed(3);

  // Discrete random variable: Deficient (< Q1), Normal, Excess (> Q3) with empirical PMF.
  function pmf3(vals) {
    const s = sortedCopy(vals), q1 = quantile(s, 0.25), q3 = quantile(s, 0.75), n = vals.length;
    const c = { Deficient: 0, Normal: 0, Excess: 0 };
    for (const v of vals) { if (v > q3) c.Excess++; else if (v < q1) c.Deficient++; else c.Normal++; }
    return { n, q1, q3, counts: c, pmf: { Deficient: c.Deficient / n, Normal: c.Normal / n, Excess: c.Excess / n } };
  }

  // Bayes' rule on the annual/monsoon top-quartile events. Both thresholds come from the same rows used for conditioning.
  function bayes(data) {
    const rows = data.filter(r => isNum(r.ANNUAL) && isNum(r['Jun-Sep']));
    const qA = quantile(sortedCopy(rows.map(r => r.ANNUAL)), 0.75), qB = quantile(sortedCopy(rows.map(r => r['Jun-Sep'])), 0.75);
    const hA = rows.filter(r => r.ANNUAL > qA), hB = rows.filter(r => r['Jun-Sep'] > qB);
    const hAB = rows.filter(r => r.ANNUAL > qA && r['Jun-Sep'] > qB);
    const pA = hA.length / rows.length, pB = hB.length / rows.length, pBgivenA = hAB.length / hA.length;
    const bayesVal = pBgivenA * pA / pB, direct = hAB.length / hB.length;
    return { n: rows.length, qA, qB, pA, pB, pBgivenA, pAgivenB_bayes: bayesVal, pAgivenB_direct: direct,
             lift: direct / pA, monsoonShare: sum(rows.map(r => r['Jun-Sep'])) / sum(rows.map(r => r.ANNUAL)) };
  }

  // January vs July: median-split probability gap plus Pearson r (pooled over all subdivision-years).
  function independence(data) {
    const p = data.filter(r => isNum(r.JAN) && isNum(r.JUL)), n = p.length;
    const jan = p.map(r => r.JAN), jul = p.map(r => r.JUL);
    const jm = quantile(sortedCopy(jan), 0.5), lm = quantile(sortedCopy(jul), 0.5);
    const pJan = jan.filter(v => v > jm).length / n, pJul = jul.filter(v => v > lm).length / n;
    const pBoth = p.filter(r => r.JAN > jm && r.JUL > lm).length / n;
    return { n, jan, jul, janMed: jm, julMed: lm, pJan, pJul, pBoth, expected: pJan * pJul, gap: Math.abs(pBoth - pJan * pJul), ...pearson(jan, jul) };
  }

  const api = { isNum, sum, mean, sampleVar, quantile, sortedCopy, skewness, normalMLE, lognormalMLE, normPdf, lognormPdf,
                normCdf, tTwoSidedP, pearson, linreg, adjR2, fmtP, pmf3, bayes, independence };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Stats = api;
})(typeof window !== 'undefined' ? window : globalThis);
