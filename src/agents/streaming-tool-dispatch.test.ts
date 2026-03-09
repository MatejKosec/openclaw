import type { AgentTool, AgentToolResult } from "@mariozechner/pi-agent-core";
import { Type } from "@sinclair/typebox";
import { describe, expect, it, vi } from "vitest";
import { createStreamingToolDispatcher } from "./streaming-tool-dispatch.js";

function makeTool(
  name: string,
  executeFn: (id: string, params: unknown) => Promise<AgentToolResult<unknown>>,
): AgentTool {
  return {
    name,
    label: name,
    description: `Test tool: ${name}`,
    parameters: Type.Object({}),
    execute: executeFn as AgentTool["execute"],
  };
}

function okResult(text: string): AgentToolResult<unknown> {
  return { content: [{ type: "text", text }], details: {} };
}

describe("StreamingToolDispatcher", () => {
  it("pre-dispatches a tool call and serves it from cache", async () => {
    const executeFn = vi.fn().mockResolvedValue(okResult("hello"));
    const tool = makeTool("bash", executeFn);
    const dispatcher = createStreamingToolDispatcher([tool]);

    // Simulate toolcall_end event during streaming.
    dispatcher.onToolCallStreamed("call-1", "bash", { command: "echo hi" });

    // Wait for dispatch to settle.
    await dispatcher.barrier();

    // Wrap the tool and call execute as the SDK would.
    const wrapped = dispatcher.wrapTool(tool);
    const result = await wrapped.execute("call-1", { command: "echo hi" });

    expect(result.content).toEqual([{ type: "text", text: "hello" }]);
    expect(dispatcher.cacheHits).toBe(1);
    expect(dispatcher.dispatched).toBe(1);
    // The original execute should have been called exactly once (from eager dispatch).
    expect(executeFn).toHaveBeenCalledTimes(1);

    dispatcher.dispose();
  });

  it("falls through to original execute when not pre-dispatched", async () => {
    const executeFn = vi.fn().mockResolvedValue(okResult("fallback"));
    const tool = makeTool("read", executeFn);
    const dispatcher = createStreamingToolDispatcher([tool]);

    const wrapped = dispatcher.wrapTool(tool);
    const result = await wrapped.execute("call-2", { path: "/tmp/x" });

    expect(result.content).toEqual([{ type: "text", text: "fallback" }]);
    expect(dispatcher.cacheHits).toBe(0);
    expect(dispatcher.dispatched).toBe(0);
    // Original execute called directly (no pre-dispatch).
    expect(executeFn).toHaveBeenCalledTimes(1);

    dispatcher.dispose();
  });

  it("awaits pending dispatch if SDK runs before it resolves", async () => {
    let resolveExec!: (value: AgentToolResult<unknown>) => void;
    const executeFn = vi.fn().mockImplementation(
      () =>
        new Promise<AgentToolResult<unknown>>((resolve) => {
          resolveExec = resolve;
        }),
    );
    const tool = makeTool("bash", executeFn);
    const dispatcher = createStreamingToolDispatcher([tool]);

    // Start eager dispatch (will be pending).
    dispatcher.onToolCallStreamed("call-3", "bash", {});

    // SDK calls wrapped execute before dispatch resolves.
    const wrapped = dispatcher.wrapTool(tool);
    const resultPromise = wrapped.execute("call-3", {});

    // Resolve the eager dispatch.
    resolveExec(okResult("delayed"));

    const result = await resultPromise;
    expect(result.content).toEqual([{ type: "text", text: "delayed" }]);
    expect(dispatcher.cacheHits).toBe(1);

    dispatcher.dispose();
  });

  it("throws on cached error result", async () => {
    const executeFn = vi.fn().mockRejectedValue(new Error("tool failed"));
    const tool = makeTool("bash", executeFn);
    const dispatcher = createStreamingToolDispatcher([tool]);

    dispatcher.onToolCallStreamed("call-4", "bash", {});
    await dispatcher.barrier();

    const wrapped = dispatcher.wrapTool(tool);
    await expect(wrapped.execute("call-4", {})).rejects.toThrow("tool failed");
    expect(dispatcher.cacheHits).toBe(1);

    dispatcher.dispose();
  });

  it("does not dispatch when disposed", async () => {
    const executeFn = vi.fn().mockResolvedValue(okResult("nope"));
    const tool = makeTool("bash", executeFn);
    const dispatcher = createStreamingToolDispatcher([tool]);

    dispatcher.dispose();
    dispatcher.onToolCallStreamed("call-5", "bash", {});

    expect(dispatcher.dispatched).toBe(0);
    expect(executeFn).not.toHaveBeenCalled();
  });

  it("does not dispatch when signal is aborted", async () => {
    const executeFn = vi.fn().mockResolvedValue(okResult("nope"));
    const tool = makeTool("bash", executeFn);
    const controller = new AbortController();
    controller.abort();
    const dispatcher = createStreamingToolDispatcher([tool], controller.signal);

    dispatcher.onToolCallStreamed("call-6", "bash", {});

    expect(dispatcher.dispatched).toBe(0);
    expect(executeFn).not.toHaveBeenCalled();

    dispatcher.dispose();
  });

  it("barrier resolves immediately when no pending dispatches", async () => {
    const tool = makeTool("bash", vi.fn());
    const dispatcher = createStreamingToolDispatcher([tool]);

    // Should not hang.
    await dispatcher.barrier();

    dispatcher.dispose();
  });

  it("skips duplicate dispatch for same toolCallId", async () => {
    const executeFn = vi.fn().mockResolvedValue(okResult("once"));
    const tool = makeTool("bash", executeFn);
    const dispatcher = createStreamingToolDispatcher([tool]);

    dispatcher.onToolCallStreamed("call-7", "bash", {});
    dispatcher.onToolCallStreamed("call-7", "bash", {});

    await dispatcher.barrier();

    expect(dispatcher.dispatched).toBe(1);
    expect(executeFn).toHaveBeenCalledTimes(1);

    dispatcher.dispose();
  });
});
