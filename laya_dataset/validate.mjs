import { readdir, readFile } from "node:fs/promises";

const base = new URL("./data/", import.meta.url);
const names = (await readdir(base)).filter((name) => name.endsWith(".jsonl")).sort();
const seen = new Map();
let failed = false;
let totalRows = 0;
let totalQuestions = 0;

for (const name of names) {
  const split = name.startsWith("train-") ? "train" : name.replace(".jsonl", "");
  const text = await readFile(new URL(name, base), "utf8");
  const lines = text.split(/\r?\n/).filter(Boolean);

  for (let index = 0; index < lines.length; index++) {
    totalRows++;
    let row;
    try { row = JSON.parse(lines[index]); }
    catch (error) {
      console.error(`${name}:${index + 1}: invalid JSON: ${error}`);
      failed = true;
      continue;
    }

    for (const key of ["state", "questions", "gold"]) {
      if (typeof row[key] !== "string") {
        console.error(`${name}:${index + 1}: ${key} must be a JSON string`);
        failed = true;
      }
    }

    let state, questions, gold;
    try {
      state = JSON.parse(row.state);
      questions = JSON.parse(row.questions);
      gold = JSON.parse(row.gold);
    } catch (error) {
      console.error(`${name}:${index + 1}: nested JSON parse failed: ${error}`);
      failed = true;
      continue;
    }

    const stateKey = JSON.stringify(state);
    const oldSplit = seen.get(stateKey);
    if (oldSplit && oldSplit !== split) {
      console.error(`${name}:${index + 1}: state duplicates split ${oldSplit}`);
      failed = true;
    } else {
      seen.set(stateKey, split);
    }

    for (const [qid, q] of Object.entries(questions)) {
      totalQuestions++;
      const probs = gold[qid]?.probabilities;
      if (!probs) {
        console.error(`${name}:${index + 1}: missing gold for ${qid}`);
        failed = true;
        continue;
      }
      const keys = q.type === "noul" ? ["false", "true"] : Object.keys(q.criteria ?? {});
      for (const key of keys) {
        if (!(key in probs)) {
          console.error(`${name}:${index + 1}: missing probability ${qid}.${key}`);
          failed = true;
        }
      }
      const sum = Object.values(probs).reduce((a, b) => a + Number(b), 0);
      if (Math.abs(sum - 1) > 1e-6) {
        console.error(`${name}:${index + 1}: probabilities for ${qid} sum to ${sum}`);
        failed = true;
      }
    }
  }

  console.log(`${name}: ${lines.length} rows`);
}

console.log(`total: ${totalRows} rows, ${totalQuestions} typed questions`);
if (failed) process.exitCode = 1;
else console.log("Laya dataset validation passed.");
