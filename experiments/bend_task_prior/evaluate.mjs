import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [modulePath, configPath, outputPath] = process.argv.slice(2);
const {default: B} = await import(pathToFileURL(path.resolve(modulePath)));
const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const out = fs.openSync(outputPath, 'wx');
let id = 0;
function emit(row) { fs.writeSync(out, JSON.stringify({id: id++, ...row}) + '\n'); }
for (const mode of cfg.conditions) for (let prior = 0; prior < 16; prior++) {
  let stored = prior, effective = prior, anchor = null;
  if (mode === 'erased') stored = effective = 0;
  if (mode.startsWith('relative')) stored = effective = B.relative(prior);
  if (mode === 'relative_plus_anchor') {
    anchor = B.lookup(prior, 0); effective = B.task(stored, anchor);
  }
  if (mode === 'fixed_recode_repaired') {
    stored = B.recode(prior, cfg.fixed_recode_mask);
    effective = B.recode(stored, cfg.fixed_recode_mask);
  }
  // Each task's support example replaces the single current-task offset.
  for (let offsets = 0; offsets < 4; offsets++) {
    const targets = [B.task(prior, offsets & 1), B.task(prior, (offsets >> 1) & 1)];
    for (let c0 = 0; c0 < 4; c0++) for (let c1 = 0; c1 < 4; c1++) {
      const labels = [B.lookup(targets[0], c0), B.lookup(targets[1], c1)];
      const adapted = [B.adapt(effective, c0, labels[0]), B.adapt(effective, c1, labels[1])];
      for (let q0 = 0; q0 < 4; q0++) for (let q1 = 0; q1 < 4; q1++) {
        if (q0 === c0 || q1 === c1) continue;
        emit({kind: 'episode', mode, prior, offsets, support: [c0, c1], query: [q0, q1],
          stored, anchor, labels, adapted,
          predictions: [B.answer(effective, adapted[0], q0), B.answer(effective, adapted[1], q1)],
          protected: [B.lookup(effective, q0), B.lookup(effective, q1)]});
      }
    }
  }
}
for (let prior = 0; prior < 16; prior++) for (let context = 0; context < 4; context++) {
  for (let offset = 0; offset < 2; offset++) {
    const retained = B.relative(prior), label = B.lookup(B.task(prior, offset), context);
    const recoveredAnchor = B.adapt(retained, context, label) ^ offset;
    emit({kind: 'anchor', prior, context, offset, retained, label, recoveredAnchor,
      restoredPrior: B.task(retained, recoveredAnchor)});
  }
}
for (let t = 0; t <= cfg.certificate_max_tasks; t++) for (let prior = 0; prior < 16; prior++) {
  for (let offsets = 0; offsets < 2 ** t; offsets++) {
    const pairedPrior = B.task(prior, 1), pairedOffsets = offsets ^ (2 ** t - 1);
    const tasks = [], pairedTasks = [];
    for (let j = 0; j < t; j++) {
      tasks.push(B.task(prior, (offsets >> j) & 1));
      pairedTasks.push(B.task(pairedPrior, (pairedOffsets >> j) & 1));
    }
    emit({kind: 'transcript', t, prior, offsets, retained: B.relative(prior), tasks,
      pairedPrior, pairedOffsets, pairedRetained: B.relative(pairedPrior), pairedTasks});
  }
}
fs.closeSync(out);
console.log(JSON.stringify({rows: id}));
