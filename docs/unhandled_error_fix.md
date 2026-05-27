# Handling Unhandled Errors from Telemetry Dashboard

## Overview
When an unhandled error appears in the telemetry dashboard it typically includes:
- **Error message**
- **Stack trace**
- **Hit count**
- **Affected user count**

The common pitfall is to **fix the crash site** (e.g., add a guard in the function that panics). This only masks the symptom while the underlying invalid data continues to flow through the system, causing future crashes.

## Recommended Workflow

### 1. Do **NOT** fix at the crash site
- Adding `typeof` guards, swallowing the error, or returning fallback values merely hides the problem.
- The root cause remains and can surface elsewhere.

### 2. Trace the data flow **upwards** through the call stack
- Read the stack trace from **bottom to top**.
- For each frame, identify:
  - The data being passed and the expected shape.
  - The origin of that data (IPC message, extension API call, storage, user input, etc.).
  - Possibilities for corruption or malformed values.
- Goal: locate the **producer** of the invalid data, not the consumer that crashes.

### 3. When the producer cannot be identified from the stack alone
- The sending side may be in a different process. In this case:
  1. **Enrich the error** at the consumer:
     - Include the type of the invalid data.
     - Add a truncated representation of the value.
     - Mention the operation/command that received it.
  2. **Do not swallow the error** – let it propagate so telemetry still captures it, but now with enough context to pinpoint the sender.
  3. Optionally add the same enrichment to the low‑level validation function that throws, ensuring the telemetry always sees the problematic payload.

### 4. When the producer **is** identifiable
- Fix the producer directly:
  - Validate or sanitise data before sending it over IPC, persisting it, or passing it to APIs.
  - Ensure correct (de)serialization – e.g., a `UriComponents` object should remain an object, not be stringified.

## Example Walk‑through
```
at _validateUri (uri.ts)       ← validation throws
at new Uri (uri.ts)            ← constructor
at URI.revive (uri.ts)         ← revive assumes valid UriComponents
at SomeChannel.call (ipc.ts)   ← IPC handler receives arg from another process
```
- **Wrong fix**: Add a guard in `URI.revive` that returns `undefined` for non‑objects. This silences the error but the caller still expects a valid URI and will later fail.
- **Right fix (producer unknown)**: Enrich the error at the IPC handler and in `_validateUri`:
```ts
// IPC handler – early validation
function reviveUri(data: UriComponents | URI | undefined | null, context: string): URI {
    if (data && typeof data !== 'object') {
        throw new Error(`[Channel] Invalid URI data for '${context}': type=${typeof data}, value=${String(data).substring(0,100)}`);
    }
    // …
}

// Validation – include offending value
throw new Error(`[UriError] Scheme contains illegal characters. scheme:"${ret.scheme.substring(0,50)}" (len:${ret.scheme.length})`);
```
- **Right fix (producer known)**: Locate the call that sends a stringified URI instead of a `UriComponents` object and correct it to pass the proper structure.

## Understanding Error Construction Before Fixing
1. **Search for the error construction** – locate the class name or a unique substring of the message.
2. **Read the construction code** to discover:
   - Trigger conditions (thresholds, validation checks, state assertions).
   - Error categories or sub‑types.
   - Meaning of any embedded parameters (ratios, flags, counts).
   - Whether the error is actionable or merely informative.
3. Use this knowledge to decide the appropriate fix strategy rather than guessing from the message alone.

### Example: Listener Leak Errors
```ts
const kind = topCount / listenerCount > 0.3 ? 'dominated' : 'popular';
const error = new ListenerLeakError(kind, message, topStack);
```
- **Dominated** (`ratio > 30%`): One code path is responsible for most listeners – fix disposal in that path.
- **Popular** (`ratio ≤ 30%`): Many paths each add a few listeners – the top stack shown is *not* the root cause. Investigate the emitter and overall subscription pattern; do **not** remove caching/pooling code without evidence.

## Guidelines Summary
- **Enrich**, don’t swallow, error messages.
- **Truncate** any user‑controlled values to avoid leaking PII.
- Keep shared utility functions (e.g., `URI.revive`) unchanged for all callers; fix at the specific call site or the data producer.
- Run the relevant unit tests after changes.
- Verify the build (`cargo build` / `cargo test`) passes before marking the work complete.

---
*This document codifies the standard process for handling unhandled errors gathered from the telemetry dashboard.*
