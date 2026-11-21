#!/usr/bin/env node
// Copies the repo's git hooks (.github/pre-commit and .github/pre-push) into this clone's hooks
// folder. `npm install` runs it (the `prepare` script), so a hook edited on main reaches a clone
// at its next install.
//
// A copy, not `core.hooksPath` on the tracked folder: with that, any branch could add a hook such
// as `post-checkout`, and it would run on a plain `git checkout` of that branch.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const git = (...args) => execFileSync('git', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
const tryGit = (...args) => {
  try {
    return git(...args);
  } catch {
    return '';
  }
};

// Not a git clone (an npm tarball, a copied folder): nothing to install.
if (!tryGit('rev-parse', '--git-dir')) process.exit(0);

// Clones set up before this script pointed git at the tracked folder: undo that.
if (tryGit('config', '--local', 'core.hooksPath') === '.github') git('config', '--local', '--unset', 'core.hooksPath');
if (tryGit('config', 'core.hooksPath')) {
  console.log('install-hooks: core.hooksPath is set, so the repo hooks were left out. Run them by hand: sh .github/pre-commit');
  process.exit(0);
}

const root = git('rev-parse', '--show-toplevel');
const hooks = path.resolve(process.cwd(), git('rev-parse', '--git-path', 'hooks'));
fs.mkdirSync(hooks, { recursive: true });
for (const name of ['pre-commit', 'pre-push']) {
  const target = path.join(hooks, name);
  fs.copyFileSync(path.join(root, '.github', name), target);
  fs.chmodSync(target, 0o755);
}
console.log(`install-hooks: pre-commit and pre-push copied to ${hooks}`);
