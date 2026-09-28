// Run: node test_stats.js   (prints every number quoted in README.md, computed by the same code the pages use)
const fs = require('fs'), S = require('./stats.js');
function parse(file) {
  const L = fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean), h = L[0].split(',');
  return L.slice(1).map(l => { const p = l.split(','), o = {}; h.forEach((k, i) => { o[k] = k === 'SUBDIVISION' ? p[i] : (p[i] === 'NA' || p[i] === '' ? null : +p[i]); }); return o; });
}
const d = parse('rainfall in india 1901-2015.csv');
const ann = d.map(r => r.ANNUAL).filter(S.isNum), mon = d.map(r => r['Jun-Sep']).filter(S.isNum);
const out = { rawRows: d.length, annualN: ann.length, mean: S.mean(ann), median: S.quantile(S.sortedCopy(ann), .5), std: Math.sqrt(S.sampleVar(ann)),
  pmf: S.pmf3(ann), monsoonN: mon.length, normal: S.normalMLE(mon), lognormal: S.lognormalMLE(mon), skew: S.skewness(mon),
  bayes: S.bayes(d) };
out.pNegative = S.normCdf(0, out.normal.mu, out.normal.sigma);
const ind = S.independence(d); delete ind.jan; delete ind.jul; out.indep = ind;
const k = d.filter(r => r.SUBDIVISION.includes('KERALA') && S.isNum(r.YEAR) && S.isNum(r.ANNUAL));
out.kerala = S.linreg(k.map(r => r.YEAR), k.map(r => r.ANNUAL)); out.keralaN = k.length;
console.log(JSON.stringify(out, null, 1));
