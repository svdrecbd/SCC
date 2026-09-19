// This bridge only serializes inputs/outputs; all VM semantics run in Bend.
import fs from 'node:fs';
import readline from 'node:readline';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const [moduleFile, inputFile, outputFile] = process.argv.slice(2);
const {default: vm} = await import(pathToFileURL(path.resolve(moduleFile)));
const natFields = ['pc', 'steps', 'reads', 'writes', 'copies'];
function tape(ops) {
  let t = {$: 'End'};
  for (let i = ops.length - 1; i >= 0; --i) t = {$: 'Code', op: ops[i], tail: t};
  return t;
}
function state(input) {
  const s = {$: 'S', ...input};
  for (const k of natFields) s[k] = BigInt(s[k]);
  return s;
}
function plain(s) {
  const result = {...s};
  delete result.$;
  for (const k of natFields) result[k] = Number(result[k]);
  return result;
}
const fd = fs.openSync(outputFile, 'wx');
let count = 0;
for await (const line of readline.createInterface({input: fs.createReadStream(inputFile)})) {
  const c = JSON.parse(line);
  let results;
  if (c.kind === 'single') {
    results = plain(vm.run(BigInt(c.fuel), tape(c.program), state(c.state), c.x, c.role));
  } else if (c.kind === 'trajectory') {
    let mem = c.mem;
    results = [];
    for (const [i, [x, role]] of c.inputs.entries()) {
      const p = c.programs ? c.programs[i] : c.program;
      const s = plain(vm.run(BigInt(c.fuel), tape(p), vm.initial(mem), x, role));
      results.push(s);
      mem = s.mem;
    }
  } else throw new Error(`unknown case kind: ${c.kind}`);
  fs.writeSync(fd, JSON.stringify({id: c.id, result: results}) + '\n');
  count++;
}
fs.closeSync(fd);
console.log(JSON.stringify({evaluated: count, backend: 'checked Bend -> JavaScript'}));
