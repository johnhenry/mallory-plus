# mallory-mcp

An [MCP](https://modelcontextprotocol.io) server exposing the Mallory family's
math engines as agent-callable tools: exact symbolic computation
(mallory-math's `Symbolic` CAS) plus guarded numeric tensor/linalg/stats
conveniences. Fast-starting, pure JS, no kernel, no license check — run it
anywhere Node runs.

```bash
# stdio transport — works with any MCP host
npx mallory-mcp

# e.g. with Claude Code:
claude mcp add mallory -- npx mallory-mcp
```

## Tools (v1)

| Tool | What it does |
|---|---|
| `symbolic_parse` | Parse/normalize an expression; returns text, LaTeX, free variables |
| `symbolic_simplify` | Exact symbolic simplification |
| `symbolic_differentiate` | d/d(variable), exact |
| `symbolic_integrate` | Indefinite (symbolic) or definite (numeric, with `lower`/`upper`) |
| `symbolic_solve` | Solve `lhs = rhs` or zero-equals form for a variable (polynomials ≤ degree 6) |
| `symbolic_evaluate` | Numeric evaluation with variable bindings |
| `linalg_solve` | Solve A·x = b (LU with partial pivoting) |
| `tensor_pipeline` | Chain of ops from a **closed** op table over a matrix — not code execution |
| `stats_summary` | count/mean/std/min/max/median for a list of numbers |

## Security posture (v1 scope, by design)

- **No arbitrary code execution anywhere.** Expression strings go through
  `Symbolic.parse` — a closed parser over a fixed grammar, not `eval`.
  Numeric tools dispatch over a fixed op table; unknown ops are rejected
  with the list of what's allowed.
- **Stateless per call.** No sessions, no notebook state, nothing persists
  between calls.
- **Size-capped inputs** (1e6 elements) so a mis-typed request errors fast
  instead of exhausting host memory.
- Tool failures return MCP `isError` results with the underlying message —
  never protocol-level exceptions.

## What v1 does NOT do

- No matrix arguments *between* pipeline steps (e.g. matmul with a second
  operand) — the pipeline is unary-chain only.
- No ODE solving, Taylor series, or step-by-step derivations, though
  mallory-math supports them — kept out of v1 to hold the tool list small
  and legible for agents; file an issue if you want one promoted.
- No HTTP/SSE transport — stdio only.
