// Execute the checked transition kernel. The bridge supplies only enumeration.
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [modulePath, outputPath] = process.argv.slice(2);
const {default: K} = await import(pathToFileURL(modulePath));
const rows = [];
for (let mode=0; mode<2; mode++) for (let state=0; state<16; state++) {
  function visit(word, current) {
    rows.push({mode, state, word, result:current});
    if (word.length===4) return;
    for (let a=0; a<2; a++) visit([...word,a], K.step_code(mode,current,a));
  }
  visit([],state);
}
fs.writeFileSync(outputPath, JSON.stringify(rows));
console.log(JSON.stringify({records:rows.length}));
