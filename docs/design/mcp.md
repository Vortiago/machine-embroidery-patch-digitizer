# MCP and WebMCP: the web UI's tools for an AI client

Status: proposed with the fork's PR on top of `feat/webui`. Builds on
`docs/design/webui.md`; every HTTP shape, edge-case rule (E1 to E9) and the
one-job-at-a-time lock come from there unchanged.

## What it adds

An AI client can do what the operator does on the page: Slice a picture with stitch
type overrides, read what happened to each override, make the Stitch plan, check on it,
and cancel it. Two ways in, with the same four tools, and a view that draws the
results inside the chat client:

| Way in | For | Start |
|---|---|---|
| **MCP server on stdio** | desktop and terminal clients (Claude Desktop, Claude Code, any MCP client) | `python3 digitize.py mcp [--port N]` |
| **WebMCP** in the page | an agent in the browser that has the page open | `python3 digitize.py web`, then open the page |
| **MCP App** view | clients that support MCP Apps (`io.modelcontextprotocol/ui`) | comes with the stdio server |

## Tools

| Tool | Does | Returns |
|---|---|---|
| `slice` | `POST /api/picture` when `picture` names a new file (stdio only), then `POST /api/slice`; waits until the Slice ends (up to 120 s) | state, the Slice with each override's result from `regions.json` (`applied`, `refused` with the coverage %, `orphaned`, `shadowed`; `pending` without a sidecar), file paths; stdio also returns the posterized PNG as an image |
| `make_stitch_plan` | `POST /api/stitchplan`; waits up to `wait_s` (default 60 s) | the Stitch plan (stitches, size, threads, warnings, `removed_by_precheck`, `stale` per E7), file paths; stdio also returns the stitch plan PNG |
| `status` | `GET /api/status` | the same report; `note` says when the job still runs |
| `cancel` | `POST /api/cancel` | `{"cancelled": true \| false}` |

`slice` takes `hoop`, `size_mm`, `colors`, `key_background`, `overrides` and, on stdio,
`picture`, `config` (a YAML path) and `yaml_text`. A setting left out keeps its value from
the last Slice in the session (stdio) or what the page shows (WebMCP). A new picture
clears the overrides (E5). On the page the tool sets the sidebar controls first, so the
operator sees what was sliced.

## Decisions

| # | Decision | Why |
|---|---|---|
| M1 | The stdio server drives `server.create_app` in process through Flask's test client. No second job runner. | A tool call and a button press run the same checks, the same job code and the same lock (G4); the server tests already cover that code. |
| M2 | The MCP protocol part (initialize, ping, tools/list, tools/call; newline-delimited JSON-RPC 2.0) is written with the standard library, about 100 lines. No `mcp` SDK. | D8's no-new-dependency rule. The tools need only this subset. The SDK can replace it later without changing the tools. |
| M3 | `--port N` also serves the page from the same app, so the browser shows each run the agent starts. | The operator can watch and take over. One job, one state. |
| M4 | WebMCP tools live in `app.js` and press the page's own buttons; the page registers them only when `document.modelContext` (current draft) or `navigator.modelContext` (earlier builds) exists. | No change for browsers without WebMCP; no new server route; the page stays one plain JS file. |
| M5 | `slice` waits for the Slice; `make_stitch_plan` waits at most `wait_s`, then the client calls `status`. | A Slice is seconds; an export can be 85 min cold (webui.md, Measured facts). |
| M6 | Each `tools/call` runs in its own thread. | `status` and `cancel` answer while `make_stitch_plan` waits. |
| M7 | `slice`, `make_stitch_plan` and `status` name one view, `ui://digitize/view.html` (`static/mcp_app.html`, mime type `text/html;profile=mcp-app`), served by `resources/read`. | Clients without MCP Apps ignore `_meta.ui` and still get the text and pictures. |
| M8 | The view gets everything from the tool result: the Region map SVG and object kinds in `_meta["digitize/view"]` (SVGs over 2 MB are left out), pictures as image content. It loads nothing from the network and needs no CSP entries. | The view runs in a sandboxed frame that cannot reach `127.0.0.1`. |
| M9 | In the view the operator clicks a Region, picks Satin, Fill or Clear, and presses **Slice again** or **Make stitch plan**; the buttons call the tools through the client (`tools/call`). | The Region map loop of the page, inside the chat. Same rules: D3 locked kinds, E1 one override per Region. |

## Not in this step

- Pan and zoom in the MCP App view, and the picture pane beside the Region map.
- Setting overrides by Region id or colour instead of a picture point (D7 keys them by
  point).
- More than one job at a time.

## Tests

`tools/digitize/tests/test_mcp_server.py`: the protocol (initialize version, tools/list,
notifications, unknown method and tool, bad JSON) and each tool against the stand-in
scripts from `test_web_server.py` (waiting, settings carried over, E5, E7, busy, cancel,
the strict Host check with `--port`), and the view's resource and `_meta`. The WebMCP tools
were checked in headless Chromium with a stand-in `document.modelContext` against the same
stand-in scripts; the view was checked in headless Chromium inside a stand-in host frame
that answers `ui/initialize` and `tools/call` with recorded tool results.
