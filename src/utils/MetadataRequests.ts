import { countMetric, recordMetric } from './performance.js';

export interface MetadataOptions { signal?: AbortSignal; priority?: 'foreground' | 'background' }
export type MetadataInput = AbortSignal | MetadataOptions;
export function metadataOptions(input?: MetadataInput): MetadataOptions {
  return input instanceof AbortSignal ? { signal: input } : input ?? {};
}
interface Subscriber<T> { resolve(value: T): void; reject(error: unknown): void; cleanup(): void }
interface Operation<T> {
  key: string; controller: AbortController; priority: 'foreground' | 'background';
  task(signal: AbortSignal): Promise<T>; subscribers: Set<Subscriber<T>>; queuedAt: number;
}

/** Two metadata jobs, at most one background job. Consumers cancel independently. */
export class MetadataRequests<T> {
  private readonly operations = new Map<string, Operation<T>>();
  private waiting: Operation<T>[] = [];
  private active = 0;
  private background = 0;
  run(key: string, task: (signal: AbortSignal) => Promise<T>, options: MetadataOptions = {}): Promise<T> {
    options.signal?.throwIfAborted();
    let operation = this.operations.get(key);
    if (!operation) {
      operation = { key, task, priority: options.priority ?? 'foreground', controller: new AbortController(),
        subscribers: new Set(), queuedAt: performance.now() };
      this.operations.set(key, operation); this.waiting.push(operation);
    } else {
      countMetric('request.shared');
      if (options.priority !== 'background') operation.priority = 'foreground';
    }
    const shared = operation;
    const promise = new Promise<T>((resolve, reject) => {
      const abort = (): void => {
        shared.subscribers.delete(subscriber); subscriber.cleanup();
        reject(options.signal?.reason ?? new Error('Request cancelled'));
        if (!shared.subscribers.size) {
          if (this.operations.get(key) === shared) this.operations.delete(key);
          shared.controller.abort();
          this.waiting = this.waiting.filter(item => item !== shared);
          this.pump();
        }
      };
      const subscriber: Subscriber<T> = { resolve, reject,
        cleanup: () => options.signal?.removeEventListener('abort', abort) };
      shared.subscribers.add(subscriber);
      options.signal?.addEventListener('abort', abort, { once: true });
    });
    queueMicrotask(() => this.pump());
    return promise;
  }
  private pump(): void {
    while (this.active < 2) {
      let index = this.waiting.findIndex(item => item.priority === 'foreground');
      if (index < 0 && this.background < 1) index = this.waiting.findIndex(item => item.priority === 'background');
      if (index < 0) return;
      const operation = this.waiting.splice(index, 1)[0]!;
      if (operation.controller.signal.aborted) continue;
      const background = operation.priority === 'background';
      this.active++; if (background) this.background++;
      recordMetric('metadata.wait', performance.now() - operation.queuedAt);
      const started = performance.now();
      void Promise.resolve().then(() => operation.task(operation.controller.signal)).then(
        value => { for (const subscriber of operation.subscribers) subscriber.resolve(value); },
        error => { for (const subscriber of operation.subscribers) subscriber.reject(error); },
      ).finally(() => {
        recordMetric('metadata.work', performance.now() - started);
        for (const subscriber of operation.subscribers) subscriber.cleanup();
        operation.subscribers.clear();
        if (this.operations.get(operation.key) === operation) this.operations.delete(operation.key);
        this.active--; if (background) this.background--;
        this.pump();
      });
    }
  }
}
