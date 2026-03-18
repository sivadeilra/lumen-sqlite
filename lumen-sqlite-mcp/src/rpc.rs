//! `--rpc` mode: a JSON-RPC server over stdin/stdout for the VS Code webview.
//!
//! Protocol: newline-delimited JSON (NDJSON). One object per line.
//!
//!   Request:  `{"id": <any>, "method": "...", "params": {...}}`
//!   Response: `{"id": <any>, "result": <value>}`
//!         or  `{"id": <any>, "error": "message"}`
//!
//! Methods
//! -------
//!   list_files      → {files: [{path, name}], db_dir}
//!   open_database   → {alias, path, created}
//!   close_database  → {alias, status}
//!   list_databases  → {databases: [{alias, path}]}
//!   database_info   → {alias, path, tables: [{name, row_count, columns, indexes}]}
//!   query           → {columns, rows, row_count, truncated, offset}
//!   execute         → {rows_affected}

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::types::ValueRef;
use serde_json::{json, Value};

use crate::{
    escape_identifier, open_db_connection, JsonParam, OpenDatabase, ALLOWED_EXTENSIONS,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct RpcState {
    connections: Mutex<HashMap<String, OpenDatabase>>,
    db_dir: PathBuf,
    default_max_rows: usize,
    sandbox: bool,
}

impl RpcState {
    fn new(db_dir_override: Option<String>) -> Self {
        let db_dir = db_dir_override
            .or_else(|| std::env::var("LUMEN_SQLITE_DB_DIR").ok())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
            });
        let db_dir = db_dir.canonicalize().unwrap_or(db_dir);

        let default_max_rows = std::env::var("LUMEN_SQLITE_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1000);

        let sandbox = std::env::var("LUMEN_SQLITE_SANDBOX")
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        Self {
            connections: Mutex::new(HashMap::new()),
            db_dir,
            default_max_rows,
            sandbox,
        }
    }

    /// Resolve and validate a database file path (mirrors LumenSqlite::resolve_db_path).
    fn resolve_db_path(&self, path_str: &str) -> Result<PathBuf, String> {
        let p = Path::new(path_str);
        let p = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.db_dir.join(p)
        };

        let resolved = if p.exists() {
            p.canonicalize().map_err(|e| format!("Cannot resolve path: {e}"))?
        } else {
            let parent = p
                .parent()
                .ok_or_else(|| "Invalid path: no parent directory".to_string())?;
            let parent = if parent.as_os_str().is_empty() {
                self.db_dir.clone()
            } else if parent.exists() {
                parent
                    .canonicalize()
                    .map_err(|e| format!("Cannot resolve parent: {e}"))?
            } else {
                return Err(format!(
                    "Parent directory '{}' does not exist",
                    parent.display()
                ));
            };
            parent.join(p.file_name().unwrap())
        };

        let ext = resolved.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !ALLOWED_EXTENSIONS.contains(&ext) {
            return Err(format!(
                "File extension '.{ext}' is not allowed. Use one of: {}",
                ALLOWED_EXTENSIONS
                    .iter()
                    .map(|e| format!(".{e}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if self.sandbox && !resolved.starts_with(&self.db_dir) {
            return Err(format!(
                "Path '{}' is outside the allowed directory '{}'.",
                resolved.display(),
                self.db_dir.display()
            ));
        }

        Ok(resolved)
    }
}

// ---------------------------------------------------------------------------
// Value conversion
// ---------------------------------------------------------------------------

/// Convert a SQLite value reference to a JSON value, preserving type.
/// Integers → JSON number, reals → JSON number, text → JSON string,
/// NULL → JSON null, blobs → human-readable placeholder string.
fn value_ref_to_json(vr: ValueRef<'_>) -> Value {
    match vr {
        ValueRef::Null => Value::Null,
        ValueRef::Integer(i) => json!(i),
        ValueRef::Real(f) => json!(f),
        ValueRef::Text(t) => json!(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Blob(b) => json!(format!("<blob {} bytes>", b.len())),
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch(state: &RpcState, method: &str, params: &Value) -> Result<Value, String> {
    match method {
        "list_files"     => handle_list_files(state),
        "open_database"  => handle_open_database(state, params),
        "close_database" => handle_close_database(state, params),
        "list_databases" => handle_list_databases(state),
        "database_info"  => handle_database_info(state, params),
        "query"          => handle_query(state, params),
        "execute"        => handle_execute(state, params),
        other => Err(format!(
            "Unknown method: '{other}'. Valid: \
             list_files, open_database, close_database, list_databases, \
             database_info, query, execute"
        )),
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

fn handle_list_files(state: &RpcState) -> Result<Value, String> {
    let mut files: Vec<Value> = Vec::new();
    for ext in ALLOWED_EXTENSIONS {
        let pattern = state.db_dir.join(format!("*.{ext}"));
        if let Ok(entries) = glob::glob(pattern.to_string_lossy().as_ref()) {
            for entry in entries.flatten() {
                let name = entry
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();
                files.push(json!({ "path": entry.to_string_lossy(), "name": name }));
            }
        }
    }
    files.sort_by(|a, b| {
        a["name"]
            .as_str()
            .unwrap_or("")
            .cmp(b["name"].as_str().unwrap_or(""))
    });
    Ok(json!({ "files": files, "db_dir": state.db_dir.to_string_lossy() }))
}

fn handle_open_database(state: &RpcState, params: &Value) -> Result<Value, String> {
    let path_str = params["path"].as_str().ok_or("missing 'path'")?;
    let db_path = state.resolve_db_path(path_str)?;

    let alias = params["alias"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            db_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("db")
                .to_string()
        });

    let mut conns = state.connections.lock().unwrap();
    if conns.contains_key(&alias) {
        return Err(format!(
            "Database '{alias}' is already open. \
             Close it first or choose a different alias."
        ));
    }

    let creating = !db_path.exists();
    let conn = open_db_connection(&db_path)?;
    conns.insert(alias.clone(), OpenDatabase { conn, path: db_path.clone() });

    Ok(json!({
        "alias":   alias,
        "path":    db_path.to_string_lossy(),
        "created": creating,
    }))
}

fn handle_close_database(state: &RpcState, params: &Value) -> Result<Value, String> {
    let alias = params["alias"].as_str().ok_or("missing 'alias'")?;
    let mut conns = state.connections.lock().unwrap();
    if conns.remove(alias).is_some() {
        Ok(json!({ "alias": alias, "status": "closed" }))
    } else {
        Err(format!("No open database with alias '{alias}'"))
    }
}

fn handle_list_databases(state: &RpcState) -> Result<Value, String> {
    let conns = state.connections.lock().unwrap();
    let mut databases: Vec<Value> = conns
        .iter()
        .map(|(alias, db)| json!({ "alias": alias, "path": db.path.to_string_lossy() }))
        .collect();
    databases.sort_by(|a, b| {
        a["alias"]
            .as_str()
            .unwrap_or("")
            .cmp(b["alias"].as_str().unwrap_or(""))
    });
    Ok(json!({ "databases": databases }))
}

fn handle_database_info(state: &RpcState, params: &Value) -> Result<Value, String> {
    let alias = params["alias"].as_str().ok_or("missing 'alias'")?;
    let conns = state.connections.lock().unwrap();
    let db = conns
        .get(alias)
        .ok_or_else(|| format!("No open database with alias '{alias}'"))?;

    // Collect table names first.
    let table_names: Vec<String> = {
        let mut stmt = db
            .conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .map_err(|e| format!("SQL error: {e}"))?;
        stmt.query_map([], |row| row.get(0))
            .and_then(|m| m.collect())
            .map_err(|e| format!("SQL error: {e}"))?
    };

    let mut tables: Vec<Value> = Vec::new();
    for table_name in &table_names {
        let safe = escape_identifier(table_name);

        let row_count: i64 = db
            .conn
            .query_row(&format!("SELECT COUNT(*) FROM \"{safe}\""), [], |r| r.get(0))
            .unwrap_or(-1);

        let columns: Vec<Value> = {
            let mut stmt = db
                .conn
                .prepare(&format!("PRAGMA table_info(\"{safe}\")"))
                .map_err(|e| format!("PRAGMA error: {e}"))?;
            stmt.query_map([], |row| {
                Ok(json!({
                    "name":          row.get::<_, String>(1)?,
                    "type":          row.get::<_, String>(2)?,
                    "notnull":       row.get::<_, i32>(3)? != 0,
                    "default_value": row.get::<_, Option<String>>(4)?,
                    "pk":            row.get::<_, i32>(5)? != 0,
                }))
            })
            .map_err(|e| format!("{e}"))?
            .flatten()
            .collect()
        };

        let indexes: Vec<Value> = {
            let idx_list: Vec<(String, bool)> = db
                .conn
                .prepare(&format!("PRAGMA index_list(\"{safe}\")"))
                .ok()
                .and_then(|mut s| {
                    s.query_map([], |row| {
                        Ok((row.get::<_, String>(1)?, row.get::<_, i32>(2)? != 0))
                    })
                    .ok()
                    .map(|m| m.flatten().collect())
                })
                .unwrap_or_default();

            idx_list
                .into_iter()
                .map(|(idx_name, unique)| {
                    let safe_idx = escape_identifier(&idx_name);
                    let cols: Vec<String> = db
                        .conn
                        .prepare(&format!("PRAGMA index_info(\"{safe_idx}\")"))
                        .ok()
                        .and_then(|mut s| {
                            s.query_map([], |row| row.get::<_, String>(2))
                                .ok()
                                .map(|m| m.flatten().collect())
                        })
                        .unwrap_or_default();
                    json!({ "name": idx_name, "unique": unique, "columns": cols })
                })
                .collect()
        };

        tables.push(json!({
            "name":      table_name,
            "row_count": row_count,
            "columns":   columns,
            "indexes":   indexes,
        }));
    }

    Ok(json!({
        "alias":  alias,
        "path":   db.path.to_string_lossy(),
        "tables": tables,
    }))
}

fn handle_query(state: &RpcState, params: &Value) -> Result<Value, String> {
    let alias = params["alias"].as_str().ok_or("missing 'alias'")?;
    let sql = params["sql"].as_str().ok_or("missing 'sql'")?;
    let max_rows = params["max_rows"]
        .as_u64()
        .map(|n| n as usize)
        .unwrap_or(state.default_max_rows);
    let offset = params["offset"].as_u64().map(|n| n as usize).unwrap_or(0);
    let bind_params: Vec<JsonParam> = params["params"]
        .as_array()
        .map(|arr| arr.iter().cloned().map(JsonParam).collect())
        .unwrap_or_default();

    let conns = state.connections.lock().unwrap();
    let db = conns
        .get(alias)
        .ok_or_else(|| format!("No open database with alias '{alias}'"))?;

    let mut stmt = db
        .conn
        .prepare(sql)
        .map_err(|e| format!("SQL error: {e}"))?;

    let col_count = stmt.column_count();
    let columns: Vec<String> = (0..col_count)
        .map(|i| stmt.column_name(i).unwrap_or("?").to_string())
        .collect();

    // Fetch max_rows + 1 to detect truncation without a separate COUNT(*) query.
    let mapped = stmt
        .query_map(rusqlite::params_from_iter(bind_params.iter()), |row| {
            Ok((0..col_count)
                .map(|i| row.get_ref(i).map(value_ref_to_json).unwrap_or(Value::Null))
                .collect::<Vec<_>>())
        })
        .map_err(|e| format!("Query error: {e}"))?;

    let mut rows: Vec<Value> = mapped
        .flatten()
        .skip(offset)
        .take(max_rows + 1)
        .map(Value::Array)
        .collect();

    let truncated = rows.len() > max_rows;
    if truncated {
        rows.truncate(max_rows);
    }
    let row_count = rows.len();

    Ok(json!({
        "columns":   columns,
        "rows":      rows,
        "row_count": row_count,
        "truncated": truncated,
        "offset":    offset,
    }))
}

fn handle_execute(state: &RpcState, params: &Value) -> Result<Value, String> {
    let alias = params["alias"].as_str().ok_or("missing 'alias'")?;
    let sql = params["sql"].as_str().ok_or("missing 'sql'")?;
    let bind_params: Vec<JsonParam> = params["params"]
        .as_array()
        .map(|arr| arr.iter().cloned().map(JsonParam).collect())
        .unwrap_or_default();

    let conns = state.connections.lock().unwrap();
    let db = conns
        .get(alias)
        .ok_or_else(|| format!("No open database with alias '{alias}'"))?;

    let rows_affected = db
        .conn
        .execute(sql, rusqlite::params_from_iter(bind_params.iter()))
        .map_err(|e| format!("SQL error: {e}"))?;

    Ok(json!({ "rows_affected": rows_affected }))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_rpc_server(db_dir_override: Option<String>) {
    let state = RpcState::new(db_dir_override);
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    eprintln!(
        "lumen-sqlite RPC server ready (db_dir={})",
        state.db_dir.display()
    );

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let (id, method, params) = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => {
                let id = msg.get("id").cloned().unwrap_or(Value::Null);
                let method = msg
                    .get("method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let params = msg
                    .get("params")
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default()));
                (id, method, params)
            }
            Err(e) => {
                let response = json!({ "id": null, "error": format!("Invalid JSON: {e}") });
                let _ = writeln!(out, "{}", response);
                let _ = out.flush();
                continue;
            }
        };

        let response = match dispatch(&state, &method, &params) {
            Ok(result) => json!({ "id": id, "result": result }),
            Err(e)     => json!({ "id": id, "error": e }),
        };
        let _ = writeln!(out, "{}", response);
        let _ = out.flush();
    }
}
