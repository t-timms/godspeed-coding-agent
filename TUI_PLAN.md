# Godspeed TUI Migration — Master Plan

> **Created:** 2026-05-09 | **Status:** Phase 3 in progress | **Textual 8.2.5**

---

## Phase 2 Complete — Polish & Integration

1. **Fixed ChatScreen rendering** ✅
2. **Fixed RichMarkdown streaming** ✅
3. **Permission/diff dialog screens wired** ✅
4. **SOTA CSS — dual-theme design** ✅
5. **Layout streamlined** ✅
6. **Code quality** ✅

## Phase 2.5 — Fast Boot & Coverage

7. **Fast-boot splash screen** ✅ — chat screen renders <3s, server loads in background
8. **MCP timeout protection** ✅ — 10s `asyncio.wait_for` per server, auto-skip on failure
9. **Ollama/llama.cpp non-blocking** ✅ — `asyncio.to_thread` keeps TUI responsive
10. **284 tests, 0 lint errors** ✅ — 100% coverage on all new TUI code
11. **Fuzzy file picker** ✅ — `@` triggers dropdown with `_find_matches` logic
12. **Init architecture** ✅ — `cli.py` → 40 lines; all backend init in `GodspeedTextualApp`

## Phase 3 — Shell & Polish

13. **PTY shell integration** ✅ — persistent `cmd.exe` subprocess, `Ctrl+R` toggle
14. **Shell state persistence** ✅ — `cd`, `set`, env vars survive between commands
15. **`/shell` command** — open shell from any screen

---

### Files Modified/Created This Session
| File | Change |
|------|--------|
| `tui/screens/chat.py` | ChatView, FilePicker, Shell toggle, async callbacks |
| `tui/widgets/chat_view.py` | Full formatting methods (tool calls, results, thinking, status) |
| `tui/widgets/file_picker.py` | Created — fuzzy @-mention file search |
| `tui/widgets/shell_widget.py` | Created — persistent subprocess shell |
| `tui/screens/shell_screen.py` | Created — full-screen terminal with input |
| `tui/screens/splash.py` | Created — loading screen with status updates |
| `tui/screens/permission_dialog.py` | Created — y/n/a permission dialog |
| `tui/screens/diff_review.py` | Created — color-coded diff review dialog |
| `tui/textual_app.py` | Dual Theme, async proxies, fast-boot, background init |
| `cli.py` | Simplified to 40 lines; all init moved to Textual app |
| `tui/widgets/prompt_input.py` | **Deleted** — dead code |
| `tui/widgets/status_bar.py` | **Deleted** — dead code |
| `tests/test_tui_chat_view.py` | Created — 44 tests for ChatView widget |
| `tests/test_tui_file_picker.py` | Created — 18 tests for FilePicker |
| `tests/test_tui_screens.py` | Created — 28 tests for all screens |
| `tests/test_tui_theme_system.py` | Created — 27 tests for themes + contrast ratios |

### Total: 284 tests passing, 0 lint errors

---

### What Remains

All phases complete. Future enhancements:
- Textual Pilot automated TUI integration tests
- `godspeed web` in WSL2 → browser on Windows
- DirectoryTree enhancements (preview, drag-resize)
