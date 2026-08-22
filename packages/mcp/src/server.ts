/**
 * @johnhenry/math-plus-mcp (issue #45): agent-callable math tools over the Model Context
 * Protocol. The design constraints, from the issue and this family's own
 * precedents (llmtm's security-gating rule, PLAN.md non-goal 2's no-eval
 * stance):
 *
 * - NO arbitrary code execution. Every tool takes structured, typed inputs.
 *   Expression strings pass through `Symbolic.parse` -- a closed parser over
 *   a fixed grammar, not `eval`.
 * - Stateless per call (v1): no sessions, no notebook state, nothing to leak
 *   between calls. This keeps the security story trivial.
 * - Numeric tools run a CLOSED op set (see `PIPELINE_OPS`), sized-capped
 *   inputs, and return plain JSON -- an agent cannot construct anything the
 *   op table doesn't name.
 *
 * The star is symbolic math (agents want exact CAS answers and are bad at
 * producing them token-by-token); the numeric tools are conveniences over
 * tensor-core / adapter-math's linalg.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { Symbolic } from "@johnhenry/math";
import { Tensor } from "@johnhenry/math-plus-tensor-core";
import { linalg } from "@johnhenry/math-plus-adapter-math";

/** Upper bound on total elements accepted by the numeric tools -- an agent
 * typo ("make a 1e9-element tensor") should error fast, not OOM the host. */
const MAX_ELEMENTS = 1_000_000;

// ---- helpers -----------------------------------------------------------------

interface ToolResult {
  [x: string]: unknown;
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
}

function ok(payload: unknown): ToolResult {
  return { content: [{ type: "text", text: JSON.stringify(payload, null, 2) }] };
}

function err(e: unknown): ToolResult {
  const message = e instanceof Error ? e.message : String(e);
  return { content: [{ type: "text", text: `Error: ${message}` }], isError: true };
}

/** Both `Expr`-facing formats agents actually consume: plain text (re-parseable
 * by these same tools) and LaTeX (renderable). */
function exprOut(expr: Parameters<typeof Symbolic.toString>[0]): { text: string; latex: string } {
  return { text: Symbolic.toString(expr), latex: Symbolic.toLatex(expr) };
}

/** Accept either the family's zero-equals convention or an explicit
 * `lhs = rhs` equation (rewritten to `(lhs) - (rhs)` BEFORE parsing -- the
 * rewrite is textual but both halves still go through the closed parser). */
function asZeroEquals(expression: string): string {
  const parts = expression.split("=");
  if (parts.length === 1) return expression;
  if (parts.length === 2) return `(${parts[0]}) - (${parts[1]})`;
  throw new Error("expression may contain at most one '='");
}

function toNumberMatrix(rows: number[][], what: string): Tensor {
  const cols = rows[0]?.length ?? 0;
  if (rows.length === 0 || cols === 0) throw new Error(`${what} must be a non-empty 2-D array`);
  if (rows.some((r) => r.length !== cols)) throw new Error(`${what} rows must all have the same length`);
  if (rows.length * cols > MAX_ELEMENTS) throw new Error(`${what} exceeds the ${MAX_ELEMENTS}-element cap`);
  return Tensor.from(rows.flat(), { dtype: "f64" }).reshape([rows.length, cols]);
}

// ---- the closed numeric-pipeline op table -------------------------------------

type PipelineStep = { op: string; axis?: number; scalar?: number; shape?: number[] };

/** Elementwise map for the few ops Tensor doesn't expose as methods (exp/
 * abs) -- fine at MCP scale (element cap above), and still a closed set:
 * the Math.* callee is fixed per table row, never caller-supplied. */
function mapElementwise(t: Tensor, fn: (x: number) => number): Tensor {
  const c = t.contiguous();
  return Tensor.from([...(c.data as Float64Array)].map(fn), { dtype: "f64" }).reshape([...c.shape]);
}

/** The ENTIRE numeric vocabulary of `tensor_pipeline`. Adding an op means
 * adding a row here -- there is deliberately no generic escape hatch. */
const PIPELINE_OPS: Record<string, (t: Tensor, step: PipelineStep) => Tensor> = {
  sum: (t, s) => (s.axis === undefined ? t.sum() : t.sum(s.axis)),
  mean: (t, s) => (s.axis === undefined ? t.mean() : t.mean(s.axis)),
  min: (t, s) => (s.axis === undefined ? t.min() : t.min(s.axis)),
  max: (t, s) => (s.axis === undefined ? t.max() : t.max(s.axis)),
  abs: (t) => mapElementwise(t, Math.abs),
  exp: (t) => mapElementwise(t, Math.exp),
  log: (t) => t.log(),
  sqrt: (t) => t.sqrt(),
  neg: (t) => t.mul(-1),
  transpose: (t) => t.transpose(),
  reshape: (t, s) => {
    if (!s.shape) throw new Error("reshape requires a shape parameter");
    return t.reshape(s.shape);
  },
  addScalar: (t, s) => {
    if (s.scalar === undefined) throw new Error("addScalar requires a scalar parameter");
    return t.add(s.scalar);
  },
  mulScalar: (t, s) => {
    if (s.scalar === undefined) throw new Error("mulScalar requires a scalar parameter");
    return t.mul(s.scalar);
  },
};

function tensorToJson(t: Tensor): unknown {
  if (t.ndim === 0) return t.item();
  const out: unknown[] = [];
  for (let i = 0; i < (t.shape[0] as number); i++) {
    out.push(tensorToJson(t.slice({ start: i, end: i + 1 }).contiguous().reshape(t.shape.slice(1))));
  }
  return out;
}

// ---- server ------------------------------------------------------------------

export function buildServer(): McpServer {
  const server = new McpServer({ name: "@johnhenry/math-plus-mcp", version: "0.0.0" });

  server.registerTool(
    "symbolic_parse",
    {
      description:
        "Parse a math expression (e.g. 'x^2 + sin(x)*3') and return its normalized text form, LaTeX, and free variables. Use this to validate/normalize before other symbolic tools.",
      inputSchema: { expression: z.string() },
    },
    ({ expression }) => {
      try {
        const e = Symbolic.parse(expression);
        return ok({ ...exprOut(e), freeVariables: Symbolic.freeVariables(e) });
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "symbolic_simplify",
    {
      description: "Simplify a math expression symbolically (exact, not numeric). Returns text + LaTeX.",
      inputSchema: { expression: z.string() },
    },
    ({ expression }) => {
      try {
        return ok(exprOut(Symbolic.simplify(expression)));
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "symbolic_differentiate",
    {
      description: "Differentiate an expression with respect to a variable (default 'x'). Exact symbolic result as text + LaTeX.",
      inputSchema: { expression: z.string(), variable: z.string().default("x") },
    },
    ({ expression, variable }) => {
      try {
        return ok(exprOut(Symbolic.differentiate(expression, variable)));
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "symbolic_integrate",
    {
      description:
        "Integrate an expression with respect to a variable (default 'x'). Omit lower/upper for the indefinite integral (symbolic text + LaTeX, no '+C'); provide both for a definite integral (number).",
      inputSchema: {
        expression: z.string(),
        variable: z.string().default("x"),
        lower: z.number().optional(),
        upper: z.number().optional(),
      },
    },
    ({ expression, variable, lower, upper }) => {
      try {
        if (lower !== undefined !== (upper !== undefined)) {
          throw new Error("provide both lower and upper for a definite integral, or neither");
        }
        if (lower !== undefined && upper !== undefined) {
          return ok({ value: Symbolic.integrateDefinite(expression, lower, upper, variable) });
        }
        return ok(exprOut(Symbolic.integrate(expression, variable)));
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "symbolic_solve",
    {
      description:
        "Solve an equation for a variable (default 'x'). Accepts 'lhs = rhs' or the zero-equals form (an expression assumed = 0). Polynomial equations up to degree 6. Returns all solutions as text + LaTeX.",
      inputSchema: { expression: z.string(), variable: z.string().default("x") },
    },
    ({ expression, variable }) => {
      try {
        const solutions = Symbolic.solve(asZeroEquals(expression), variable);
        return ok({ solutions: solutions.map(exprOut) });
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "symbolic_evaluate",
    {
      description:
        "Numerically evaluate an expression with the given variable values, e.g. expression='x^2+y', variables={x:3,y:1} -> 10.",
      inputSchema: { expression: z.string(), variables: z.record(z.number()).default({}) },
    },
    ({ expression, variables }) => {
      try {
        return ok({ value: Symbolic.evaluate(expression, variables) });
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "linalg_solve",
    {
      description:
        "Solve the linear system A·x = b exactly (LU with partial pivoting). a: square matrix as number[][]; b: number[]. Returns x as number[].",
      inputSchema: { a: z.array(z.array(z.number())), b: z.array(z.number()) },
    },
    ({ a, b }) => {
      try {
        if (b.length > MAX_ELEMENTS) throw new Error(`b exceeds the ${MAX_ELEMENTS}-element cap`);
        const A = toNumberMatrix(a, "a");
        const B = Tensor.from(b, { dtype: "f64" });
        const x = linalg.solve(A, B);
        return ok({ x: [...(x.contiguous().data as Float64Array)] });
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "tensor_pipeline",
    {
      description:
        `Run a short pipeline of named numeric ops over a matrix (number[][]), NOT arbitrary code. Ops: ${Object.keys(PIPELINE_OPS).join(", ")}. Each step: {op, axis?, scalar?, shape?}. Returns the resulting value/array plus its shape.`,
      inputSchema: {
        data: z.array(z.array(z.number())),
        ops: z
          .array(
            z.object({
              op: z.string(),
              axis: z.number().int().optional(),
              scalar: z.number().optional(),
              shape: z.array(z.number().int()).optional(),
            }),
          )
          .max(16),
      },
    },
    ({ data, ops }) => {
      try {
        let t = toNumberMatrix(data, "data");
        for (const step of ops) {
          const fn = PIPELINE_OPS[step.op];
          if (!fn) throw new Error(`unknown op "${step.op}" -- the closed op set is: ${Object.keys(PIPELINE_OPS).join(", ")}`);
          t = fn(t, step);
          if (t.size > MAX_ELEMENTS) throw new Error(`intermediate result exceeds the ${MAX_ELEMENTS}-element cap`);
        }
        return ok({ shape: t.shape, result: tensorToJson(t) });
      } catch (e) {
        return err(e);
      }
    },
  );

  server.registerTool(
    "stats_summary",
    {
      description: "Descriptive statistics for a list of numbers: count, mean, population/sample std, min, max, median.",
      inputSchema: { values: z.array(z.number()).min(1).max(MAX_ELEMENTS) },
    },
    ({ values }) => {
      try {
        const t = Tensor.from(values, { dtype: "f64" });
        const sorted = [...values].sort((x, y) => x - y);
        const n = values.length;
        const median = n % 2 === 1 ? sorted[(n - 1) / 2] : ((sorted[n / 2 - 1] as number) + (sorted[n / 2] as number)) / 2;
        const mean = t.mean().item() as number;
        const deviations = t.sub(mean);
        const popVar = (deviations.mul(deviations).sum().item() as number) / n;
        return ok({
          count: n,
          mean,
          populationStd: Math.sqrt(popVar),
          sampleStd: n > 1 ? Math.sqrt((popVar * n) / (n - 1)) : 0,
          min: sorted[0],
          max: sorted[n - 1],
          median,
        });
      } catch (e) {
        return err(e);
      }
    },
  );

  return server;
}
