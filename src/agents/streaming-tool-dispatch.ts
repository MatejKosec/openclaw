/**
 * Streaming tool call dispatch — eagerly runs tool calls as they arrive
 * in the LLM stream (at `toolcall_end` boundaries) rather than waiting
 * for the full assistant message to complete.
 *
 * The SDK's agent-loop executes tools sequentially AFTER the stream ends.
 * This module pre-dispatches them in parallel during streaming so that
 * results are already cached when the SDK's `tool.execute()` fires.
 *
 * Barriers:
 *  - `<think>` open tag: await all pending dispatches (model's next reasoning
 *    block may depend on earlier tool results).
 *  - Stream end (message_end): await all remaining pending dispatches.
 *
 * Invariant: the LLM stream is NEVER interrupted — generation continues
 * regardless of dispatch status.
 */

import type {
  AgentTool,
  AgentToolResult,
  AgentToolUpdateCallback,
} from "@mariozechner/pi-agent-core";
import type { ImageContent, TextContent } from "@mariozechner/pi-ai";
import { validateToolArguments } from "@mariozechner/pi-ai";
import { createSubsystemLogger } from "../logging/subsystem.js";

const log = createSubsystemLogger("agent/streaming-dispatch");

/** Extract text content from a tool result's content array. */
function extractTextFromContent(content: (TextContent | ImageContent)[]): string {
  return content
    .filter((c): c is TextContent => c.type === "text")
    .map((c) => c.text)
    .join("\n");
}

/** Hash a tool call for cache lookup: name + sorted JSON args. */
function toolCallKey(toolCallId: string): string {
  return toolCallId;
}

interface PendingDispatch {
  toolCallId: string;
  toolName: string;
  promise: Promise<CachedResult>;
}

interface CachedResult {
  result: AgentToolResult<unknown>;
  isError: boolean;
}

export interface StreamingToolDispatcher {
  /**
   * Called when a `toolcall_end` event arrives during streaming.
   * Immediately fires the tool execution in the background.
   */
  onToolCallStreamed(toolCallId: string, toolName: string, args: Record<string, unknown>): void;

  /**
   * Called when a `<think>` open tag is detected during streaming.
   * Awaits all pending dispatched tool calls — the model's reasoning
   * block may depend on their results.
   */
  barrier(): Promise<void>;

  /**
   * Wrap a tool's execute function so it checks the pre-computed cache first.
   * If a result was pre-dispatched, returns it instantly; otherwise falls
   * through to the original execute.
   */
  wrapTool<T extends AgentTool>(tool: T): T;

  /**
   * Clean up: cancel any pending dispatches and clear the cache.
   */
  dispose(): void;

  /** Number of tool calls that were served from the pre-dispatch cache. */
  readonly cacheHits: number;

  /** Number of tool calls dispatched during streaming. */
  readonly dispatched: number;
}

export function createStreamingToolDispatcher(
  tools: AgentTool[],
  signal?: AbortSignal,
): StreamingToolDispatcher {
  const toolsByName = new Map<string, AgentTool>();
  for (const tool of tools) {
    toolsByName.set(tool.name, tool);
  }

  // Cache of pre-computed results indexed by toolCallId.
  const cache = new Map<string, CachedResult>();
  // Pending dispatches that haven't resolved yet.
  const pending = new Map<string, PendingDispatch>();

  let cacheHits = 0;
  let dispatched = 0;
  let disposed = false;

  function onToolCallStreamed(
    toolCallId: string,
    toolName: string,
    args: Record<string, unknown>,
  ): void {
    if (disposed || signal?.aborted) {
      return;
    }
    const key = toolCallKey(toolCallId);
    // Already dispatched or cached — skip.
    if (cache.has(key) || pending.has(key)) {
      return;
    }

    const tool = toolsByName.get(toolName);
    if (!tool) {
      log.debug(`streaming dispatch: tool not found: ${toolName}`);
      return;
    }

    dispatched++;
    log.debug(`streaming dispatch: firing ${toolName} (id=${toolCallId})`);

    const promise = (async (): Promise<CachedResult> => {
      try {
        // Validate and coerce args (matches what the SDK does before calling execute).
        const toolCall = {
          type: "toolCall" as const,
          id: toolCallId,
          name: toolName,
          arguments: args,
        };
        const validatedArgs = validateToolArguments(tool, toolCall);
        // Execute with the abort signal so cancellation propagates.
        const result = await tool.execute(toolCallId, validatedArgs, signal);
        return { result, isError: false };
      } catch (err) {
        return {
          result: {
            content: [{ type: "text", text: err instanceof Error ? err.message : String(err) }],
            details: {},
          },
          isError: true,
        };
      }
    })();

    const entry: PendingDispatch = { toolCallId, toolName, promise };
    pending.set(key, entry);

    // When resolved, move from pending to cache.
    void promise.then((cached) => {
      pending.delete(key);
      if (!disposed) {
        cache.set(key, cached);
        log.debug(`streaming dispatch: cached result for ${toolName} (id=${toolCallId})`);
      }
    });
  }

  async function barrier(): Promise<void> {
    if (pending.size === 0) {
      return;
    }
    log.debug(`streaming dispatch: barrier — awaiting ${pending.size} pending dispatch(es)`);
    const promises = [...pending.values()].map((p) => p.promise);
    await Promise.allSettled(promises);
    log.debug("streaming dispatch: barrier cleared");
  }

  function wrapTool<T extends AgentTool>(tool: T): T {
    const originalExecute = tool.execute;
    const wrapped: T = {
      ...tool,
      execute: async (
        toolCallId: string,
        params: unknown,
        execSignal?: AbortSignal,
        onUpdate?: AgentToolUpdateCallback,
      ) => {
        const key = toolCallKey(toolCallId);
        const cached = cache.get(key);
        if (cached) {
          cache.delete(key);
          cacheHits++;
          log.debug(`streaming dispatch: cache hit for ${tool.name} (id=${toolCallId})`);
          // If the cached result was an error, throw so the SDK marks it as error.
          if (cached.isError) {
            const errorText = extractTextFromContent(cached.result.content);
            throw new Error(errorText || "Pre-dispatched tool call failed");
          }
          return cached.result;
        }

        // Check if still pending (unlikely but possible if SDK runs before dispatch resolves).
        const pendingEntry = pending.get(key);
        if (pendingEntry) {
          const result = await pendingEntry.promise;
          cache.delete(key);
          pending.delete(key);
          cacheHits++;
          log.debug(`streaming dispatch: waited for pending ${tool.name} (id=${toolCallId})`);
          if (result.isError) {
            const errorText = extractTextFromContent(result.result.content);
            throw new Error(errorText || "Pre-dispatched tool call failed");
          }
          return result.result;
        }

        // No pre-dispatch — fall through to original execution.
        return originalExecute.call(tool, toolCallId, params, execSignal, onUpdate);
      },
    };
    return wrapped;
  }

  function dispose(): void {
    disposed = true;
    cache.clear();
    pending.clear();
  }

  return {
    onToolCallStreamed,
    barrier,
    wrapTool,
    dispose,
    get cacheHits() {
      return cacheHits;
    },
    get dispatched() {
      return dispatched;
    },
  };
}
