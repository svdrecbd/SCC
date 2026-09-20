import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [modulePath, configPath, outputPath] = process.argv.slice(2);
const {default: B} = await import(pathToFileURL(path.resolve(modulePath)));
const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
const targets = cfg.queries.map(q => {
  let table = 0;
  for (let z = 0; z < 16; z++) table |= B.parity(z, q) << z;
  return table;
});
const out = fs.openSync(outputPath, 'wx');
for (let encoder = 0; encoder < cfg.encoders; encoder++) {
  const correct = targets.map(t => B.correct(encoder, t));
  const bucket_correct = targets.map(t => B.bucket_correct(encoder, t));
  fs.writeSync(out, JSON.stringify({encoder, correct, bucket_correct,
    population: B.pop16(encoder)}) + '\n');
}
fs.closeSync(out);
console.log(JSON.stringify({rows: cfg.encoders, queries: cfg.queries, targets}));
