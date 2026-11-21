#!/usr/bin/env node
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const errors = [];

function git(args, input) {
  return execFileSync('git', args, { cwd: root, encoding: 'utf8', input, maxBuffer: 64 * 1024 * 1024 });
}

/**
 * A hard cap of 5,000 tokens (bytes / 4) on every file in the index — what a
 * commit would contain, so the pre-commit hook and CI judge the same bytes. An
 * agent reads a file whole; past this size it pays for a file before reading
 * any of the code that matters to it.
 *
 * Exempt: lockfiles, data, generated and binary files (nobody authors them),
 * and GitHub workflows, whose jobs must share one file to depend on each other.
 */
const MAX_FILE_TOKENS = 5000;
const BYTES_PER_TOKEN = 4;
const MAX_FILE_BYTES = MAX_FILE_TOKENS * BYTES_PER_TOKEN;

const TOKEN_LIMIT_IGNORE = [
  /^package-lock\.json$/,
  /^ml\/uv\.lock$/,
  /(?:^|\/)dist\//,
  /^public\/data\//,
  /^test\/fixtures\//,
  /^\.github\/workflows\//,
  /\.(?:woff2?|ttf|otf|png|jpg|jpeg|gif|ico|webp|pdf|wasm|svg|bin|npz|pt|task)$/i,
];

function splitGuidanceFor(relPath) {
  if (relPath.endsWith('.py')) {
    return (
      '  How to split a Python module:\n' +
      '  1. Separate constants/tables, pure helpers, and the main class or entry point into sibling modules in ml/.\n' +
      '  2. Import from the concrete module — no facade that re-exports.\n' +
      '  3. Add the new file to modal_train.py when the remote image imports it.'
    );
  }
  if (/\.tsx?$/.test(relPath)) {
    return (
      '  How to split a TypeScript file:\n' +
      '  1. Extract cohesive components, hooks, or pure helpers into sibling files.\n' +
      '  2. Import from the concrete module — no index.ts barrels.'
    );
  }
  if (relPath.endsWith('.md')) {
    return (
      '  How to split a doc:\n' +
      '  1. Move whole sections into a companion named after the parent, leaving a one-line pointer.\n' +
      '  2. Fix #anchor links that pointed at a moved section.'
    );
  }
  return '  Break the file into smaller, single-responsibility files under 20,000 bytes.';
}

/** Every path in the index with the size of its indexed blob, in two git calls. */
function indexedFileSizes() {
  const entries = git(['ls-files', '-s', '-z'])
    .split('\0')
    .filter(Boolean)
    .map((line) => {
      const [meta, relPath] = line.split('\t');
      return { relPath, sha: meta.split(' ')[1] };
    });
  const sizes = git(
    ['cat-file', '--batch-check=%(objectsize)'],
    entries.map((e) => e.sha).join('\n') + '\n',
  )
    .trim()
    .split('\n')
    .map(Number);
  return entries.map((e, i) => ({ relPath: e.relPath, bytes: sizes[i] }));
}

function checkFileTokenLimit() {
  for (const { relPath, bytes } of indexedFileSizes()) {
    if (TOKEN_LIMIT_IGNORE.some((re) => re.test(relPath))) continue;
    const tokens = Math.ceil(bytes / BYTES_PER_TOKEN);
    if (tokens > MAX_FILE_TOKENS) {
      errors.push(
        `[file-size] ${relPath} is ${tokens.toLocaleString()} tokens (${bytes.toLocaleString()} bytes), exceeding the ${MAX_FILE_TOKENS.toLocaleString()}-token (${MAX_FILE_BYTES.toLocaleString()}-byte) hard cap (tokens = bytes / 4).\n` +
          splitGuidanceFor(relPath),
      );
    }
  }
}

/** Every `npm run <script>` the instruction docs mention must exist. */
function checkDocsToScripts() {
  const scripts = new Set(
    Object.keys(JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')).scripts || {}),
  );
  for (const doc of ['AGENTS.md', 'docs/CI-TRIAGE.md']) {
    const content = fs.readFileSync(path.join(root, doc), 'utf8');
    for (const match of content.matchAll(/npm run ([a-zA-Z0-9:_-]+)/g)) {
      if (!scripts.has(match[1])) errors.push(`[docs] ${doc} references missing npm script: ${match[1]}`);
    }
  }
}

/** Reports the first line matching `lineRe` in every indexed file whose path matches `pathRe`. */
function checkLines(pathRe, lineRe, report) {
  for (const { relPath } of indexedFileSizes()) {
    if (!pathRe.test(relPath)) continue;
    const content = fs.readFileSync(path.join(root, relPath), 'utf8');
    const hit = content.split('\n').findIndex((line) => lineRe.test(line));
    if (hit >= 0) errors.push(report(`${relPath}:${hit + 1}`));
  }
}

/** No skipped tests in the suite (the vitest equivalent of jest/no-disabled-tests). */
function checkSkippedTests() {
  checkLines(
    /^test\/.*\.test\.tsx?$/,
    /\b(?:it|test|describe)\.(?:skip|todo)\b|\bx(?:it|test|describe)\(/,
    (at) => `[tests] ${at} commits a skipped test. Fix it or delete it.`,
  );
}

/** No HTML built from strings in the app: the public demo takes URL parameters, so text goes in as text. */
function checkHtmlSinks() {
  checkLines(
    /^src\/.*\.tsx?$/,
    /\.(?:inner|outer)HTML\s*=|insertAdjacentHTML|dangerouslySetInnerHTML|document\.write/,
    (at) => `[html] ${at} builds HTML from a string. Use textContent or React elements.`,
  );
}

checkFileTokenLimit();
checkDocsToScripts();
checkSkippedTests();
checkHtmlSinks();

if (errors.length > 0) {
  console.error('Repo policy checks failed:\n');
  for (const err of errors) console.error(`- ${err}`);
  process.exit(1);
}

console.log('Repo policy checks passed.');
