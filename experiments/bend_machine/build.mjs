// Pinned Bend API adapter: check the complete book before emitting any code.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [toolchain, source, output] = process.argv.slice(2);
if (!toolchain || !source) throw new Error('usage: build.mjs BEND_ROOT SOURCE [OUTPUT]');
const Bend = await import(pathToFileURL(path.join(toolchain, 'bend2/bend.ts')));
const Comp = await import(pathToFileURL(path.join(toolchain, 'bend2/comp.ts')));
let stage = 'load';
try {
  const book = Bend.book_nil();
  const seen = new Map();
  await Bend.book_load(book, path.resolve(source), '', seen);
  if (path.basename(source) === 'PROOF.bend' &&
      !seen.has(fs.realpathSync(path.join(path.dirname(source), 'LAWS.bend')))) {
    throw new Error('PROOF.bend must import its LAWS.bend');
  }
  stage = 'typecheck';
  Bend.book_valid(book);
  stage = 'completeness';
  if (book.hols + book.open !== 0) throw new Error('unfilled holes or laws');
  const unsafe = Object.entries(book.tlds).filter(([k, v]) =>
    v.$ === 'Def' && (v.u === true || k.includes('~'))).map(([k]) => k);
  if (unsafe.length) throw new Error(`unsafe definitions: ${unsafe.join(', ')}`);
  stage = 'emit';
  if (output) fs.writeFileSync(output, Comp.js_lib(book, ['initial', 'run'], ['initial', 'run']));
  console.log(JSON.stringify({checked: true, source, output: output ?? null,
    definitions: book.order.length, holes: book.hols, open: book.open, unsafe,
    imports: [...seen.keys()]}));
} catch (e) {
  const error = e?.$ === 'Err' ? Bend.err_show(e) : String(e);
  console.error(error);
  console.log(JSON.stringify({checked: false, stage, source, error}));
  process.exitCode = 1;
}
