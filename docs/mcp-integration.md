# Godspeed MCP Integration

## 1) What Godspeed MCP mode does

Godspeed MCP mode turns Godspeed into a secure execution layer for any MCP-compatible agent: every tool call is permission-checked, audited in a hash-chained log, and secret-redacted before results are returned.

## 2) How it works

`Any MCP Agent -> MCP Protocol -> Godspeed MCP Server -> Permission Engine + Audit Trail + Built-in Tools`

Both CLI and MCP modes use the same rules and audit model; MCP adds caller attribution so you can distinguish external clients in logs.

## 3) Installation

```bash
pip install godspeed-coding-agent
# or
uv tool install godspeed-coding-agent
```

Then initialize defaults:

```bash
godspeed init
```

## 4) Starting the server

Start MCP stdio mode:

```bash
godspeed serve
```

Startup and shutdown status is written to stderr:

- `Godspeed MCP server ready`
- `Godspeed MCP server shutdown`

To verify it is running, connect an MCP client and list tools. A healthy server responds with discovered tool definitions and schemas.

## 5) Connecting an MCP client (generic)

Any MCP client needs:

- A process command to launch
- Stdio transport
- Command: `godspeed serve`

On connect, the client discovers Godspeed's registered built-in tools and their JSON schemas.

## 6) Client configuration examples

### Generic MCP client (stdio pattern)

```json
{
  "name": "godspeed",
  "transport": "stdio",
  "command": "godspeed",
  "args": ["serve"]
}
```

Clients that follow the common `mcpServers` settings convention use the
same shape in their own settings file:

```json
{
  "mcpServers": {
    "godspeed": {
      "command": "godspeed",
      "args": ["serve"]
    }
  }
}
```

## 7) Verifying security is working

Inspect audit logs after a session:

```bash
godspeed audit verify
```

Denied calls appear with `denied: true`, tool metadata, caller attribution, and reason. MCP-originated entries include caller identity such as `mcp_client:unknown` when no client metadata is available.

## 8) Permission configuration for MCP use

MCP mode uses the same `settings.yaml` permission model as CLI mode.

- Same `permissions.deny`, `permissions.allow`, `permissions.ask`
- Same dangerous-command checks
- Same audit and redaction behavior

Recommended baseline for MCP deployments:

- Keep deny rules for secrets (`.env`, keys, certs, cloud creds)
- Restrict `allow` to explicit safe command families
- Keep broad shell access under `ask` or deny by default
- Keep audit enabled and verify chains regularly
