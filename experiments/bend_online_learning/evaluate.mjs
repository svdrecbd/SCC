import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [modulePath, configPath, outputPath] = process.argv.slice(2);
const {default: B} = await import(pathToFileURL(path.resolve(modulePath)));
const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const out = fs.openSync(outputPath, 'wx');
let id = 0;
for (const mode of cfg.modes) {
  const countMode = ['count_repair', 'complemented_count', 'complemented_count_repaired', 'repair_after_four'].includes(mode);
  for (let t = 0; t <= cfg.horizon; t++) for (let history = 0; history < 2 ** t; history++) {
    let zero = 1, one = 1, count = 16, calls = 0;
    for (let step = 1; step <= t; step++) {
      let observed = (history >> (step - 1)) & 1;
      if (mode === 'input_erased_boundary') observed = 0;
      if (countMode) {
        if (mode !== 'repair_after_four' || step > cfg.reset_after) {
          if (mode.startsWith('complemented')) observed = 1 - observed;
          count = B.advance_count(count, observed);
        }
      } else if (mode !== 'neutralized_update') {
        zero = B.advance_weight(zero, 0, observed);
        one = B.advance_weight(one, 1, observed);
        calls += 2;
      }
      if (mode === 'reset_after_four' && step === cfg.reset_after) { zero = 1; one = 1; }
    }
    const native_pred = countMode ? B.predict_count(count) : B.predict(zero, one);
    const pred = mode === 'complemented_count_repaired' ? B.predict_count(B.invert(count)) : native_pred;
    fs.writeSync(out, JSON.stringify({id: id++, mode, t, history,
      state: countMode ? [count] : [zero, one], native_pred, pred,
      permission: mode === 'live_allow' ? 1 : 1 - pred, likelihood_calls: calls}) + '\n');
  }
}
fs.closeSync(out);
console.log(JSON.stringify({rows: id}));
