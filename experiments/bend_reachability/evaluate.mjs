// Execute the checked Bend kernel; no graph-solving logic in this bridge.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const [modulePath, configPath, outputPath] = process.argv.slice(2);
const {default: G} = await import(pathToFileURL(path.resolve(modulePath)));
const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const fd = fs.openSync(outputPath, 'wx');
let id = 0;
function emit(row) { fs.writeSync(fd, JSON.stringify({id: id++, ...row}) + '\n'); }
for (let g = 0; g < cfg.graphs; g++) {
  for (let s = 0; s < cfg.vertices; s++) for (let t = 0; t < cfg.vertices; t++) {
    if (s === t) continue;
    const mapped = G.relabel(g, s, t);
    const distance = G.distance(mapped);
    for (const answer of cfg.output_classes)
      emit({kind: 'point', g, s, t, answer, mapped, distance, verdict: G.decode(answer)});
  }
}
for (const scenario of cfg.scenarios) {
  const {name, mode, encoding, decoder, live_removed} = scenario;
  for (let g = 0; g < cfg.graphs; g++) {
    const raw = G.predict(mode, g, ...encoding);
    emit({kind: 'useful', name, g, raw, repaired: G.repair(raw, ...decoder)});
  }
  for (let g = 0; g < cfg.graphs; g++) {
    for (let s = 0; s < cfg.vertices; s++) for (let t = 0; t < cfg.vertices; t++) {
      if (s === t) continue;
      const mapped = G.relabel(g, s, t);
      const raw = G.predict(mode, mapped, ...encoding);
      const repaired = G.repair(raw, ...decoder);
      emit({kind: 'protected', name, g, s, t, mapped, raw, repaired,
        recovered: G.decode(repaired), live: G.live(live_removed, repaired)});
    }
  }
}
fs.closeSync(fd);
console.log(JSON.stringify({records: id, backend: 'checked Bend JavaScript CPU'}));
