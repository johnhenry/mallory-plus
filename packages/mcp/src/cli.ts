#!/usr/bin/env node
/**
 * `npx mallory-mcp` / `mallory-mcp`: run the server over stdio -- the
 * transport every MCP host (Claude Code/Desktop, `claude mcp add`, etc.)
 * speaks natively. No flags in v1: the tool set is fixed and stateless.
 */
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { buildServer } from "./server.ts";

const server = buildServer();
await server.connect(new StdioServerTransport());
