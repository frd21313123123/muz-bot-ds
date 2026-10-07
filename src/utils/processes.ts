import { spawn, type ChildProcess } from 'node:child_process';
import { trackProcess } from './performance.js';

const terminating = new WeakMap<ChildProcess, Promise<void>>();
/** Release IPC immediately, terminate descendants on Windows, and bound shutdown. */
export function terminateProcess(child: ChildProcess): Promise<void> {
  const previous = terminating.get(child);
  if (previous) return previous;
  const promise = new Promise<void>(resolve => {
    let helper: ChildProcess | null = null;
    let settled = false;
    let childClosed = false, helperFinished = process.platform !== 'win32' || !child.pid;
    const kill = (process: ChildProcess, signal?: NodeJS.Signals): void => { try { process.kill(signal); } catch { /* IPC still closes; timeout bounds teardown. */ } };
    const releasePipes = (): void => { child.stdin?.destroy(); child.stdout?.destroy(); child.stderr?.destroy(); };
    const finish = (): void => {
      if (settled) return;
      settled = true; clearTimeout(timer); child.off('close', closed);
      releasePipes(); resolve();
    };
    const complete = (): void => { if (childClosed && helperFinished) finish(); };
    const closed = (): void => { childClosed = true; complete(); };
    const fallback = (): void => { if (child.exitCode === null) kill(child); };
    const timer = setTimeout(() => { if (helper) kill(helper); kill(child, 'SIGKILL'); finish(); }, 5000);
    timer.unref(); child.once('close', closed);
    if (child.exitCode !== null || child.signalCode !== null) { finish(); return; }
    if (process.platform === 'win32' && child.pid) {
      helper = spawn('taskkill.exe', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
      trackProcess(helper);
      helper.once('error', () => { helperFinished = true; fallback(); complete(); });
      helper.once('exit', code => { helperFinished = true; if (code !== 0) fallback(); complete(); });
    } else fallback();
    // A grandchild must not keep the parent alive through inherited pipe handles.
    releasePipes();
  });
  terminating.set(child, promise);
  return promise;
}
