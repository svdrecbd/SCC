// Reuse the pinned, fully checked Bend compiler approach of LN-162.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const [root, source, output] = process.argv.slice(2);
const Bend = await import(pathToFileURL(path.resolve(root, 'bend2/bend.ts')));
const Comp = await import(pathToFileURL(path.resolve(root, 'bend2/comp.ts')));
let stage = 'load';
try {
  const book = Bend.book_nil(), seen = new Map();
  await Bend.book_load(book, path.resolve(source), '', seen);
  stage = 'typecheck';
  Bend.book_valid(book);
  stage = 'completeness';
  if (book.hols + book.open) throw new Error('unfinished proof');
  const unsafe = Object.entries(book.tlds).filter(([k, t]) =>
    t.$ === 'Def' && (t.u === true || k.includes('~'))).map(([k]) => k);
  if (unsafe.length) throw new Error('unsafe definitions');
  stage = 'emit';
  const names = ['lookup', 'task', 'relative', 'adapt', 'answer', 'recode'];
  if (output) fs.writeFileSync(output, Comp.js_lib(book, names, names));
  console.log(JSON.stringify({checked: true, holes: book.hols, open: book.open,
    unsafe, imports: [...seen.keys()]}));
} catch (e) {
  const error = e?.$ === 'Err' ? Bend.err_show(e) : String(e);
  console.error(error);
  console.log(JSON.stringify({checked: false, stage, error}));
  process.exitCode = 1;
}
