//! lumen-sqlite — An MCP server for structured data storage using SQLite.
//!
//! This server exposes tools that let a language model create, query, and manage
//! SQLite databases through the Model Context Protocol.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use csv as csv_crate;
use regex::Regex;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::schemars::JsonSchema;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use rusqlite::types::{ToSqlOutput, Value as SqlValue, ValueRef};
use rusqlite::{Connection, MAIN_DB};
use serde::Deserialize;
use tracing::info;

mod copilot_compat;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// An open database connection with its resolved file path.
struct OpenDatabase {
    conn: Connection,
    path: PathBuf,
}

/// Server state shared across all tool invocations.
pub struct LumenSqlite {
    /// Tool router for MCP dispatch.
    tool_router: ToolRouter<Self>,
    /// Open database connections keyed by alias.
    connections: Mutex<HashMap<String, OpenDatabase>>,
    /// Base directory for resolving relative database paths.
    db_dir: PathBuf,
    /// Maximum rows returned by execute_sql.
    default_max_rows: usize,
    /// Maximum rows for export_query_csv.
    default_export_max_rows: usize,
    /// When true, all paths must resolve under db_dir.
    sandbox: bool,
}

/// Allowed file extensions for database files.
const ALLOWED_EXTENSIONS: &[&str] = &["db", "sqlite", "sqlite3"];

impl LumenSqlite {
    pub fn new() -> Self {
        let db_dir = std::env::var("LUMEN_SQLITE_DB_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let db_dir = db_dir.canonicalize().unwrap_or_else(|_| db_dir.clone());

        let default_max_rows = std::env::var("LUMEN_SQLITE_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(500);

        let default_export_max_rows = std::env::var("LUMEN_SQLITE_EXPORT_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);

        let sandbox = std::env::var("LUMEN_SQLITE_SANDBOX")
            .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        Self {
            tool_router: Self::tool_router(),
            connections: Mutex::new(HashMap::new()),
            db_dir,
            default_max_rows,
            default_export_max_rows,
            sandbox,
        }
    }

    /// Resolve and validate a database file path.
    fn resolve_db_path(&self, path_str: &str) -> Result<PathBuf, String> {
        let p = Path::new(path_str);
        let p = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.db_dir.join(p)
        };

        // Canonicalize if the file already exists; otherwise canonicalize the parent.
        let resolved = if p.exists() {
            p.canonicalize()
                .map_err(|e| format!("Cannot resolve path: {e}"))?
        } else {
            let parent = p
                .parent()
                .ok_or_else(|| "Invalid path: no parent directory".to_string())?;
            let parent = if parent.exists() {
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

        // Check extension
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

        // Sandbox check
        if self.sandbox && !resolved.starts_with(&self.db_dir) {
            return Err(format!(
                "Path '{}' is outside the allowed directory '{}'. \
                 Sandbox mode is enabled (LUMEN_SQLITE_SANDBOX=1).",
                resolved.display(),
                self.db_dir.display()
            ));
        }

        Ok(resolved)
    }

    /// Resolve a file path for CSV/TSV I/O (no extension restriction).
    fn resolve_file_path(&self, path_str: &str) -> Result<PathBuf, String> {
        let p = Path::new(path_str);
        let p = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.db_dir.join(p)
        };

        // For files that already exist, canonicalize fully.
        // For new output files, canonicalize the parent directory.
        let resolved = if p.exists() {
            p.canonicalize()
                .map_err(|e| format!("Cannot resolve path: {e}"))?
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

        // Sandbox check
        if self.sandbox && !resolved.starts_with(&self.db_dir) {
            return Err(format!(
                "Path '{}' is outside the allowed directory '{}'. \
                 Sandbox mode is enabled (LUMEN_SQLITE_SANDBOX=1).",
                resolved.display(),
                self.db_dir.display()
            ));
        }

        Ok(resolved)
    }
}

// ---------------------------------------------------------------------------
// SQL value conversion helpers
// ---------------------------------------------------------------------------

/// A JSON value that can be passed as a SQLite bound parameter.
struct JsonParam(serde_json::Value);

impl rusqlite::ToSql for JsonParam {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(match &self.0 {
            serde_json::Value::Null => ToSqlOutput::Owned(SqlValue::Null),
            serde_json::Value::Bool(b) => ToSqlOutput::Owned(SqlValue::Integer(*b as i64)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ToSqlOutput::Owned(SqlValue::Integer(i))
                } else {
                    ToSqlOutput::Owned(SqlValue::Real(n.as_f64().unwrap_or(0.0)))
                }
            }
            serde_json::Value::String(s) => ToSqlOutput::Owned(SqlValue::Text(s.clone())),
            other => ToSqlOutput::Owned(SqlValue::Text(other.to_string())),
        })
    }
}

fn value_ref_to_string(vr: ValueRef<'_>) -> String {
    match vr {
        ValueRef::Null => String::new(),
        ValueRef::Integer(i) => i.to_string(),
        ValueRef::Real(f) => f.to_string(),
        ValueRef::Text(t) => String::from_utf8_lossy(t).into_owned(),
        ValueRef::Blob(b) => format!("<blob {} bytes>", b.len()),
    }
}

/// Validate that a name is a safe SQL identifier.
fn validate_identifier(name: &str, kind: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("Invalid {kind}: empty string is not allowed."));
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(format!(
            "Invalid {kind}: {name:?}. Must start with a letter or underscore."
        ));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!(
            "Invalid {kind}: {name:?}. Must match [A-Za-z_][A-Za-z0-9_]*"
        ));
    }
    Ok(())
}

/// Escape a SQL identifier for use inside double-quotes.
fn escape_identifier(name: &str) -> String {
    name.replace('"', "\"\"")
}

/// Return the CSV delimiter byte to use for a file.
/// If `delimiter` is explicitly given, parse it (accepts single ASCII char or `\t`).
/// Otherwise, auto-detect from the file extension: `.tsv` → tab, everything else → comma.
fn parse_delimiter(delimiter: &Option<String>, path: &Path) -> Result<u8, String> {
    if let Some(d) = delimiter {
        // Accept the two-character escape sequence "\t" as a tab
        if d == r"\t" || d == "\t" {
            return Ok(b'\t');
        }
        let bytes = d.as_bytes();
        if bytes.len() == 1 {
            return Ok(bytes[0]);
        }
        return Err(format!(
            "delimiter must be a single character (or \\t for tab), got: {d:?}"
        ));
    }
    // Auto-detect from extension
    if path.extension().and_then(|e| e.to_str()) == Some("tsv") {
        Ok(b'\t')
    } else {
        Ok(b',')
    }
}

/// Guess the best SQLite column type by sampling a slice of string values.
/// Returns "INTEGER", "REAL", or "TEXT".
fn infer_sqlite_type(vals: &[&str]) -> &'static str {
    let non_empty: Vec<&&str> = vals.iter().filter(|v| !v.is_empty()).collect();
    if non_empty.is_empty() {
        return "TEXT";
    }
    if non_empty.iter().all(|v| v.parse::<i64>().is_ok()) {
        return "INTEGER";
    }
    if non_empty.iter().all(|v| v.parse::<f64>().is_ok()) {
        return "REAL";
    }
    "TEXT"
}

/// Map a serde_json::Value to the SQLite column type name it represents.
fn json_value_to_sqlite_type(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            if let Some(n) = v.as_i64().or_else(|| v.as_u64().map(|u| u as i64)) {
                let _ = n;
                "INTEGER"
            } else {
                "REAL"
            }
        }
        _ => "TEXT",
    }
}

/// Shared implementation for import_json and import_json_file.
fn import_json_objects(
    connections: &Mutex<HashMap<String, OpenDatabase>>,
    alias: &str,
    table: &str,
    if_exists: &str,
    col_names: &[String],
    arr: &[serde_json::Map<String, serde_json::Value>],
) -> String {
    // Infer types from first non-null value per column
    let col_types: Vec<&'static str> = col_names
        .iter()
        .map(|col| {
            for obj in arr.iter().take(200) {
                if let Some(v) = obj.get(col) {
                    if !v.is_null() {
                        return json_value_to_sqlite_type(v);
                    }
                }
            }
            "TEXT"
        })
        .collect();

    let conns = connections.lock().unwrap();
    let db = match conns.get(alias) {
        Some(d) => d,
        None => return LumenSqlite::no_such_alias(alias, &conns),
    };

    let escaped_table = escape_identifier(table);
    let col_defs: String = col_names
        .iter()
        .zip(col_types.iter())
        .map(|(c, t)| format!("\"{}\" {t}", escape_identifier(c)))
        .collect::<Vec<_>>()
        .join(", ");
    let placeholders: String = (0..col_names.len())
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let insert_sql = format!("INSERT INTO \"{escaped_table}\" VALUES ({placeholders})");

    let result: rusqlite::Result<()> = (|| {
        if if_exists == "replace" {
            db.conn
                .execute_batch(&format!("DROP TABLE IF EXISTS \"{escaped_table}\""))?;
        } else if if_exists == "error" {
            let exists: bool = db.conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                [table],
                |row| row.get::<_, i64>(0),
            )? > 0;
            if exists {
                return Err(rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                    Some(format!(
                        "Table '{table}' already exists. Use if_exists='replace' to overwrite or if_exists='append' to add rows."
                    )),
                ));
            }
        }
        db.conn.execute_batch(&format!(
            "CREATE TABLE IF NOT EXISTS \"{escaped_table}\" ({col_defs})"
        ))?;
        db.conn.execute_batch("BEGIN TRANSACTION")?;
        let mut stmt = db.conn.prepare(&insert_sql)?;
        for obj in arr {
            let vals: Vec<serde_json::Value> = col_names
                .iter()
                .map(|c| obj.get(c).cloned().unwrap_or(serde_json::Value::Null))
                .collect();
            let row_refs: Vec<JsonParam> = vals.into_iter().map(JsonParam).collect();
            let row_sql_refs: Vec<&dyn rusqlite::ToSql> =
                row_refs.iter().map(|p| p as &dyn rusqlite::ToSql).collect();
            stmt.execute(row_sql_refs.as_slice())?;
        }
        drop(stmt);
        db.conn.execute_batch("COMMIT")?;
        Ok(())
    })();

    match result {
        Ok(()) => format!(
            "Imported {} rows into '{table}' ({} columns: {}).",
            arr.len(),
            col_names.len(),
            col_names.join(", ")
        ),
        Err(e) => {
            let _ = db.conn.execute_batch("ROLLBACK");
            format!("Import error: {e}")
        }
    }
}

// ---------------------------------------------------------------------------
// Tool parameter types
// ---------------------------------------------------------------------------

#[derive(Deserialize, JsonSchema)]
pub struct OpenDatabaseParams {
    /// Path to the .db / .sqlite / .sqlite3 file.
    /// Relative paths are resolved against the server's base directory.
    pub path: String,
    /// Short alias for this database (defaults to file stem).
    pub alias: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct AliasParams {
    /// The alias of the database.
    pub alias: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct BackupDatabaseParams {
    /// The alias of the source database.
    pub alias: String,
    /// Path for the backup file (.db / .sqlite / .sqlite3).
    pub destination: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct CreateReadmeParams {
    /// The alias of the database.
    pub alias: String,
    /// A one-sentence description of the database's purpose.
    pub purpose: String,
    /// Optional map of table names to descriptions.
    pub tables: Option<HashMap<String, String>>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExecuteSqlParams {
    /// The alias of the database to query.
    pub alias: String,
    /// The SQL statement to execute.
    pub sql: String,
    /// Optional list of bind parameters (for parameterized queries using ? placeholders).
    pub params: Option<Vec<serde_json::Value>>,
    /// Maximum rows to return per page (default: from env or 500).
    pub max_rows: Option<usize>,
    /// Row offset for pagination (default: 0). If the response says more rows are available,
    /// call again with offset incremented by the number of rows returned.
    pub offset: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExecuteScriptParams {
    /// The alias of the database.
    pub alias: String,
    /// SQL script containing one or more statements separated by semicolons.
    pub script: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExplainQueryParams {
    /// The alias of the database.
    pub alias: String,
    /// The SQL query to explain.
    pub sql: String,
    /// Optional bind parameters.
    pub params: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize, JsonSchema)]
pub struct BulkInsertParams {
    /// The alias of the database.
    pub alias: String,
    /// The table to insert into.
    pub table: String,
    /// List of column names, e.g. ["name", "age", "city"].
    pub columns: Vec<String>,
    /// List of rows, each row being a list of values matching the columns.
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Conflict resolution: "error" (default), "replace" (INSERT OR REPLACE),
    /// "ignore" (INSERT OR IGNORE — skips rows that violate a constraint).
    pub conflict: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ImportCsvParams {
    /// The alias of the database.
    pub alias: String,
    /// CSV-formatted text with a header row followed by data rows.
    pub csv_text: String,
    /// Target table name. Defaults to "imported".
    pub table: Option<String>,
    /// What to do if the table already exists: "append" (default) or "replace".
    pub if_exists: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExportQueryCsvParams {
    /// The alias of the database.
    pub alias: String,
    /// A SELECT statement.
    pub sql: String,
    /// Optional bind parameters.
    pub params: Option<Vec<serde_json::Value>>,
    /// Maximum rows to export per page (default: 10000).
    pub max_rows: Option<usize>,
    /// Row offset for pagination (default: 0).
    pub offset: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct TableParams {
    /// The alias of the database.
    pub alias: String,
    /// The name of the table.
    pub table: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct DropTableParams {
    /// The alias of the database.
    pub alias: String,
    /// The name of the table to drop.
    pub table: String,
    /// Set to true to actually drop the table. Default false (preview only).
    pub confirm: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
pub struct RenameTableParams {
    /// The alias of the database.
    pub alias: String,
    /// Current table name.
    pub old_name: String,
    /// New table name.
    pub new_name: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct InspectCsvFileParams {
    /// Path to the CSV or TSV file to inspect. Absolute or relative to LUMEN_SQLITE_DB_DIR.
    pub path: String,
    /// Column delimiter. "," for CSV (default), "\t" for TSV. Single character.
    pub delimiter: Option<String>,
    /// Number of data rows to sample for type inference (default: 5).
    pub sample_rows: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ImportCsvFileParams {
    /// The alias of the open database to import into.
    pub alias: String,
    /// Path to the CSV or TSV file. Absolute or relative to LUMEN_SQLITE_DB_DIR.
    pub path: String,
    /// Target table name. Required when has_headers is false; defaults to the file stem otherwise.
    pub table: Option<String>,
    /// Whether the first row is a header row with column names (default: true).
    pub has_headers: Option<bool>,
    /// Column names to use when has_headers is false. Must match the number of columns in the file.
    pub columns: Option<Vec<String>>,
    /// Column delimiter. "," for CSV (default), "\t" for TSV. Single character.
    pub delimiter: Option<String>,
    /// What to do if the table already exists: "append" (default) or "replace".
    pub if_exists: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExportQueryCsvFileParams {
    /// The alias of the open database.
    pub alias: String,
    /// A SELECT statement to export.
    pub sql: String,
    /// Optional bind parameters for the query.
    pub params: Option<Vec<serde_json::Value>>,
    /// Path to write the output file. Absolute or relative to LUMEN_SQLITE_DB_DIR.
    pub path: String,
    /// Whether to emit a header row with column names (default: true).
    pub emit_headers: Option<bool>,
    /// Column delimiter. "," for CSV (default), "\t" for TSV. Single character.
    pub delimiter: Option<String>,
    /// Maximum rows to export (default: 10000). Set 0 for unlimited.
    pub max_rows: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ImportJsonParams {
    /// The alias of the open database.
    pub alias: String,
    /// A JSON array of objects to import. All objects should share the same keys.
    pub json_text: String,
    /// Target table name (default: "imported").
    pub table: Option<String>,
    /// What to do if the table already exists: "error" (default), "replace", "append".
    pub if_exists: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ImportJsonFileParams {
    /// The alias of the open database.
    pub alias: String,
    /// Path to the JSON file. Absolute or relative to LUMEN_SQLITE_DB_DIR.
    pub path: String,
    /// Target table name (default: file stem).
    pub table: Option<String>,
    /// File format: "array" (default, a JSON array of objects) or "ndjson" (one JSON object per line).
    pub format: Option<String>,
    /// What to do if the table already exists: "error" (default), "replace", "append".
    pub if_exists: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExportQueryJsonFileParams {
    /// The alias of the open database.
    pub alias: String,
    /// A SELECT statement to export.
    pub sql: String,
    /// Optional bind parameters for the query.
    pub params: Option<Vec<serde_json::Value>>,
    /// Path to write the output file. Absolute or relative to LUMEN_SQLITE_DB_DIR.
    pub path: String,
    /// Output format: "array" (default, a JSON array of objects) or "ndjson" (one JSON object per line).
    pub format: Option<String>,
    /// Maximum rows to export (default: 10000). Set 0 for unlimited.
    pub max_rows: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExportQueryMarkdownParams {
    /// The alias of the open database.
    pub alias: String,
    /// A SELECT statement to render.
    pub sql: String,
    /// Optional bind parameters for the query.
    pub params: Option<Vec<serde_json::Value>>,
    /// Maximum rows to return per page (default: from env or 500).
    pub max_rows: Option<usize>,
    /// Row offset for pagination (default: 0).
    pub offset: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct TableStatsParams {
    /// The alias of the database.
    pub alias: String,
    /// The table to profile.
    pub table: String,
    /// Optional subset of columns to include. Defaults to all columns.
    pub columns: Option<Vec<String>>,
}

#[derive(Deserialize, JsonSchema)]
pub struct AttachDatabaseParams {
    /// The alias of the already-open database whose connection will run ATTACH.
    pub alias: String,
    /// Path to the SQLite database file to attach (absolute or relative to db_dir).
    pub path: String,
    /// Schema name to use in SQL (e.g. "other" → SELECT * FROM "other".my_table).
    pub schema_name: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct DetachDatabaseParams {
    /// The alias of the database connection from which to detach.
    pub alias: String,
    /// The schema name to detach (as given to attach_database).
    pub schema_name: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct CreateIndexParams {
    /// The alias of the database.
    pub alias: String,
    /// The table to index.
    pub table: String,
    /// Columns to include in the index, in order.
    pub columns: Vec<String>,
    /// Index name. Auto-generated as idx_table_col1_col2 if not provided.
    pub index_name: Option<String>,
    /// Whether to create a UNIQUE index (default: false).
    pub unique: Option<bool>,
    /// Optional WHERE clause for a partial index (e.g. "status = 'active'").
    pub where_clause: Option<String>,
    /// Use IF NOT EXISTS (default: false).
    pub if_not_exists: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
pub struct DropIndexParams {
    /// The alias of the database.
    pub alias: String,
    /// The name of the index to drop.
    pub index_name: String,
    /// Set to true to actually drop. Default false shows a preview.
    pub confirm: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
pub struct AddColumnParams {
    /// The alias of the database.
    pub alias: String,
    /// The table to alter.
    pub table: String,
    /// Name of the new column.
    pub column: String,
    /// SQLite column type (default: TEXT). Use INTEGER, REAL, BLOB, or TEXT.
    pub col_type: Option<String>,
    /// Whether the column is NOT NULL (default: false). Requires default_value.
    pub not_null: Option<bool>,
    /// Default value expression (e.g. "0", "'unknown'", "datetime('now')").
    pub default_value: Option<String>,
}

// ---------------------------------------------------------------------------
// MCP Tools
// ---------------------------------------------------------------------------

#[tool_router]
impl LumenSqlite {
    #[tool(
        description = "Open or create a SQLite database file and assign it an alias (defaults to the file stem). Creates the file if it does not exist."
    )]
    async fn open_database(&self, params: Parameters<OpenDatabaseParams>) -> String {
        let params = params.0;
        let db_path = match self.resolve_db_path(&params.path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let alias = params.alias.unwrap_or_else(|| {
            db_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("db")
                .to_string()
        });

        let mut conns = self.connections.lock().unwrap();
        if conns.contains_key(&alias) {
            return format!(
                "Database '{alias}' is already open. \
                 Close it first or choose a different alias."
            );
        }

        let creating = !db_path.exists();
        let conn = match Connection::open(&db_path) {
            Ok(c) => c,
            Err(e) => return format!("Failed to open database: {e}"),
        };

        // Enable WAL mode and foreign keys
        let _ = conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;");

        // Register REGEXP scalar function: text REGEXP pattern
        // Returns 1 if pattern matches anywhere in text, 0 otherwise.
        // NULL input yields NULL. Invalid patterns return an error.
        conn.create_scalar_function(
            "regexp",
            2,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8
                | rusqlite::functions::FunctionFlags::SQLITE_DETERMINISTIC,
            |ctx| {
                let pattern: Option<String> = ctx.get(0)?;
                let text: Option<String> = ctx.get(1)?;
                match (pattern, text) {
                    (Some(pat), Some(txt)) => {
                        let re = Regex::new(&pat).map_err(|e| {
                            rusqlite::Error::UserFunctionError(Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidInput,
                                e.to_string(),
                            )))
                        })?;
                        Ok(re.is_match(&txt) as i64)
                    }
                    _ => Ok(0i64),
                }
            },
        )
        .map_err(|e| format!("Failed to register regexp UDF: {e}"))
        .unwrap_or(());

        let verb = if creating {
            "Created and opened"
        } else {
            "Opened"
        };
        info!("{verb} database '{alias}' at {}", db_path.display());
        conns.insert(
            alias.clone(),
            OpenDatabase {
                conn,
                path: db_path.clone(),
            },
        );
        format!("{verb} database '{alias}' at {}", db_path.display())
    }

    #[tool(description = "Close an open database connection.")]
    async fn close_database(&self, params: Parameters<AliasParams>) -> String {
        let alias = params.0.alias;
        let mut conns = self.connections.lock().unwrap();
        if conns.remove(&alias).is_some() {
            info!("Closed database '{alias}'");
            format!("Closed database '{alias}'.")
        } else {
            Self::no_such_alias(&alias, &conns)
        }
    }

    #[tool(description = "List all currently open database aliases.")]
    async fn list_databases(&self) -> String {
        let conns = self.connections.lock().unwrap();
        if conns.is_empty() {
            "(none)".to_string()
        } else {
            let mut aliases: Vec<&str> = conns.keys().map(|s| s.as_str()).collect();
            aliases.sort();
            aliases.join("\n")
        }
    }

    #[tool(
        description = "List .db files in the data directory, including ones not currently open. Useful for discovering databases from prior sessions."
    )]
    async fn list_database_files(&self) -> String {
        let mut files: Vec<PathBuf> = Vec::new();
        for ext in ALLOWED_EXTENSIONS {
            let pattern = self.db_dir.join(format!("*.{ext}"));
            if let Ok(entries) = glob::glob(pattern.to_string_lossy().as_ref()) {
                for entry in entries.flatten() {
                    files.push(entry);
                }
            }
        }
        files.sort();

        if files.is_empty() {
            return format!("No database files found in {}", self.db_dir.display());
        }

        let mut lines = vec![
            format!("Database files in {}:", self.db_dir.display()),
            String::new(),
            format!("{:<40} {:>10} {}", "File", "Size", "Modified"),
            "-".repeat(70),
        ];

        for f in &files {
            if let Ok(meta) = f.metadata() {
                let size = meta.len();
                let size_str = if size < 1024 * 1024 {
                    format!("{:.1} KB", size as f64 / 1024.0)
                } else {
                    format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
                };
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| {
                        let secs = t.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
                        // Format as YYYY-MM-DD HH:MM using chrono
                        use chrono::{DateTime, TimeZone, Utc};
                        let dt: DateTime<Utc> = Utc.timestamp_opt(secs as i64, 0).single()?;
                        Some(dt.format("%Y-%m-%d %H:%M").to_string())
                    })
                    .unwrap_or_else(|| "?".to_string());
                let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("?");
                lines.push(format!("{:<40} {:>10} {}", name, size_str, mtime));
            }
        }
        lines.join("\n")
    }

    #[tool(
        description = "Get a comprehensive summary of an open database: _readme contents, all tables with row counts and column schemas. Use this instead of calling list_tables + describe_table + SELECT COUNT(*) repeatedly."
    )]
    async fn database_info(&self, params: Parameters<AliasParams>) -> String {
        let alias = params.0.alias;
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let mut lines = vec![format!("Database: {alias}"), "=".repeat(60)];

        // _readme contents
        let has_readme: bool = db
            .conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='_readme'",
                [],
                |_| Ok(true),
            )
            .unwrap_or(false);

        if has_readme {
            lines.push(String::new());
            lines.push("## _readme".to_string());
            if let Ok(mut stmt) = db
                .conn
                .prepare("SELECT key, value FROM _readme ORDER BY key")
            {
                if let Ok(rows) = stmt.query_map([], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                }) {
                    for row in rows.flatten() {
                        lines.push(format!("  {}: {}", row.0, row.1));
                    }
                }
            }
        }

        // All tables
        let tables: Vec<String> = {
            let mut stmt = match db
                .conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            {
                Ok(s) => s,
                Err(e) => return format!("Error reading tables: {e}"),
            };
            stmt.query_map([], |row| row.get::<_, String>(0))
                .map(|mapped| mapped.flatten().collect())
                .unwrap_or_default()
        };

        if tables.is_empty() {
            lines.push(String::new());
            lines.push("(no tables)".to_string());
            return lines.join("\n");
        }

        lines.push(String::new());
        lines.push("## Tables".to_string());

        for table in &tables {
            if table == "sqlite_sequence" {
                continue;
            }
            let safe = escape_identifier(table);

            let count: i64 = db
                .conn
                .query_row(&format!("SELECT COUNT(*) FROM \"{safe}\""), [], |r| {
                    r.get(0)
                })
                .unwrap_or(-1);
            let count_str = if count < 0 {
                "?".to_string()
            } else {
                count.to_string()
            };

            lines.push(String::new());
            lines.push(format!("### {table} ({count_str} rows)"));

            if let Ok(mut stmt) = db.conn.prepare(&format!("PRAGMA table_info(\"{safe}\")")) {
                if let Ok(cols) = stmt.query_map([], |row| {
                    Ok((
                        row.get::<_, String>(1)?,         // name
                        row.get::<_, String>(2)?,         // type
                        row.get::<_, i32>(3)?,            // notnull
                        row.get::<_, Option<String>>(4)?, // default
                        row.get::<_, i32>(5)?,            // pk
                    ))
                }) {
                    for col in cols.flatten() {
                        let (name, col_type, notnull, default_val, pk) = col;
                        let mut parts = vec![format!("  {name}")];
                        if !col_type.is_empty() {
                            parts.push(col_type);
                        }
                        if pk != 0 {
                            parts.push("PK".to_string());
                        }
                        if notnull != 0 {
                            parts.push("NOT NULL".to_string());
                        }
                        if let Some(d) = default_val {
                            parts.push(format!("DEFAULT {d}"));
                        }
                        lines.push(parts.join(" "));
                    }
                }
            }

            // Indexes on this table
            let indexes: Vec<(String, i32)> = {
                let mut stmt = db
                    .conn
                    .prepare(&format!("PRAGMA index_list(\"{safe}\")"))
                    .ok();
                stmt.as_mut()
                    .and_then(|s| {
                        s.query_map([], |row| {
                            Ok((
                                row.get::<_, String>(1)?, // name
                                row.get::<_, i32>(2)?,    // unique
                            ))
                        })
                        .ok()
                        .map(|m| m.flatten().collect())
                    })
                    .unwrap_or_default()
            };
            for (idx_name, unique) in &indexes {
                let safe_idx = escape_identifier(idx_name);
                let cols: Vec<String> = {
                    let mut stmt = db
                        .conn
                        .prepare(&format!("PRAGMA index_info(\"{safe_idx}\")"))
                        .ok();
                    stmt.as_mut()
                        .and_then(|s| {
                            s.query_map([], |row| row.get::<_, String>(2))
                                .ok()
                                .map(|m| m.flatten().collect())
                        })
                        .unwrap_or_default()
                };
                let unique_str = if *unique != 0 { "UNIQUE " } else { "" };
                lines.push(format!(
                    "  {unique_str}INDEX {idx_name} ({})",
                    cols.join(", ")
                ));
            }
        }

        lines.push(String::new());
        lines.join("\n")
    }

    #[tool(
        description = "Safely copy an open database to a new file using SQLite's backup API. Safe to call while the database is in use."
    )]
    async fn backup_database(&self, params: Parameters<BackupDatabaseParams>) -> String {
        let BackupDatabaseParams { alias, destination } = params.0;
        let dest_path = match self.resolve_db_path(&destination) {
            Ok(p) => p,
            Err(e) => return e,
        };

        if dest_path.exists() {
            return format!(
                "Destination '{}' already exists. Choose a different path or remove it first.",
                dest_path.display()
            );
        }

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        match db.conn.backup(MAIN_DB, &dest_path, None) {
            Ok(()) => {
                let size_kb = dest_path
                    .metadata()
                    .map(|m| m.len() as f64 / 1024.0)
                    .unwrap_or(0.0);
                info!("Backed up '{alias}' to {}", dest_path.display());
                format!(
                    "Backed up '{alias}' to {} ({size_kb:.1} KB).",
                    dest_path.display()
                )
            }
            Err(e) => {
                let _ = std::fs::remove_file(&dest_path);
                format!("Backup error: {e}")
            }
        }
    }

    #[tool(
        description = "Rebuild the database file to reclaim free space (VACUUM). After many DELETE operations, the database file doesn't shrink until VACUUM is run."
    )]
    async fn compact_database(&self, params: Parameters<AliasParams>) -> String {
        let alias = params.0.alias;
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let size_before = db.path.metadata().map(|m| m.len()).unwrap_or(0);
        match db.conn.execute_batch("VACUUM") {
            Err(e) => format!("Compact error: {e}"),
            Ok(()) => {
                let size_after = db.path.metadata().map(|m| m.len()).unwrap_or(0);
                let before_kb = size_before as f64 / 1024.0;
                let after_kb = size_after as f64 / 1024.0;
                if size_before > size_after {
                    let saved_kb = (size_before - size_after) as f64 / 1024.0;
                    format!(
                        "Compacted '{alias}': {before_kb:.1} KB → {after_kb:.1} KB (saved {saved_kb:.1} KB)."
                    )
                } else {
                    format!("Compacted '{alias}': no space to reclaim ({after_kb:.1} KB).")
                }
            }
        }
    }

    #[tool(
        description = "Create or replace a _readme table that self-documents the database. Call immediately after creating a new database. Do NOT call for databases from external sources."
    )]
    async fn create_readme(&self, params: Parameters<CreateReadmeParams>) -> String {
        let CreateReadmeParams {
            alias,
            purpose,
            tables,
        } = params.0;
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let result: rusqlite::Result<usize> = (|| {
            db.conn.execute_batch(
                "DROP TABLE IF EXISTS _readme;
                 CREATE TABLE _readme (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
            )?;
            db.conn.execute(
                "INSERT INTO _readme VALUES ('purpose', ?1)",
                rusqlite::params![purpose],
            )?;
            db.conn.execute(
                "INSERT INTO _readme VALUES ('created', datetime('now'))",
                [],
            )?;
            let mut n = 2usize;
            if let Some(ref tbl_map) = tables {
                for (tbl_name, tbl_desc) in tbl_map {
                    db.conn.execute(
                        "INSERT INTO _readme VALUES (?1, ?2)",
                        rusqlite::params![format!("table:{tbl_name}"), tbl_desc],
                    )?;
                    n += 1;
                }
            }
            Ok(n)
        })();

        match result {
            Ok(n) => {
                info!("Created _readme in '{alias}'");
                format!("Created _readme table with {n} entries in '{alias}'.")
            }
            Err(e) => format!("Error creating _readme: {e}"),
        }
    }

    #[tool(
        description = "Execute a single SQL statement. SELECT returns rows; INSERT/UPDATE/DELETE/DDL returns affected row count. Use ? placeholders with params to prevent injection. Paginate with offset+max_rows — increment offset by rows returned when response says more are available. TRAP: cannot span transactions — BEGIN in one call then COMMIT in another has no effect; use execute_script for manual transactions or bulk_insert for row batches."
    )]
    async fn execute_sql(&self, params: Parameters<ExecuteSqlParams>) -> String {
        let ExecuteSqlParams {
            alias,
            sql,
            params: bind_params,
            max_rows,
            offset,
        } = params.0;
        let max_rows = max_rows.unwrap_or(self.default_max_rows);
        let offset = offset.unwrap_or(0);
        let bind_params = bind_params.unwrap_or_default();
        let json_params: Vec<JsonParam> = bind_params.into_iter().map(JsonParam).collect();

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let mut stmt = match db.conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => return format!("SQL error: {e}"),
        };

        let col_count = stmt.column_count();

        if col_count > 0 {
            // SELECT-like: returns rows
            let col_names: Vec<String> =
                stmt.column_names().iter().map(|&s| s.to_string()).collect();

            let rows_result: rusqlite::Result<Vec<Vec<String>>> = stmt
                .query_map(rusqlite::params_from_iter(json_params.iter()), |row| {
                    let vals: Vec<String> = (0..col_count)
                        .map(|i| row.get_ref(i).map(value_ref_to_string).unwrap_or_default())
                        .collect();
                    Ok(vals)
                })
                .and_then(|mapped| {
                    let mut collected = Vec::new();
                    let mut skipped = 0usize;
                    for r in mapped {
                        if skipped < offset {
                            let _ = r?;
                            skipped += 1;
                            continue;
                        }
                        collected.push(r?);
                        if collected.len() >= max_rows + 1 {
                            break;
                        }
                    }
                    Ok(collected)
                });

            match rows_result {
                Err(e) => format!("SQL error: {e}"),
                Ok(mut rows) => {
                    let has_more = rows.len() > max_rows;
                    if has_more {
                        rows.pop();
                    }
                    if rows.is_empty() {
                        return if offset > 0 {
                            format!("(No rows at offset {offset} -- past end of results.)")
                        } else {
                            "(no rows)".to_string()
                        };
                    }
                    let start = offset + 1;
                    let end = offset + rows.len();
                    let status = if has_more {
                        format!(
                            "(Rows {start}-{end}, more available. Call again with offset={end}.)"
                        )
                    } else {
                        format!("(Rows {start}-{end}.)")
                    };
                    let mut out = status;
                    out.push('\n');
                    out.push_str(&col_names.join("\t"));
                    for row in &rows {
                        out.push('\n');
                        out.push_str(&row.join("\t"));
                    }
                    out
                }
            }
        } else {
            // DML / DDL: returns affected row count
            match stmt.execute(rusqlite::params_from_iter(json_params.iter())) {
                Ok(n) => format!("{n} row(s) affected."),
                Err(e) => format!("SQL error: {e}"),
            }
        }
    }

    #[tool(
        description = "Execute multiple semicolon-separated SQL statements in one call. Use for schema setup (CREATE TABLE, CREATE INDEX) or when you need manual transaction control (BEGIN; …; COMMIT). No parameter binding — never interpolate untrusted values into the script; use execute_sql with params for that."
    )]
    async fn execute_script(&self, params: Parameters<ExecuteScriptParams>) -> String {
        let ExecuteScriptParams { alias, script } = params.0;
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };
        match db.conn.execute_batch(&script) {
            Ok(()) => "Script executed successfully.".to_string(),
            Err(e) => format!("SQL error: {e}"),
        }
    }

    #[tool(
        description = "Show how SQLite will execute a query (EXPLAIN QUERY PLAN). Use to understand whether indices are used, whether full table scans occur, and the join order."
    )]
    async fn explain_query(&self, params: Parameters<ExplainQueryParams>) -> String {
        let ExplainQueryParams {
            alias,
            sql,
            params: bind_params,
        } = params.0;
        let bind_params = bind_params.unwrap_or_default();
        let json_params: Vec<JsonParam> = bind_params.into_iter().map(JsonParam).collect();

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let explain_sql = format!("EXPLAIN QUERY PLAN {sql}");
        let mut stmt = match db.conn.prepare(&explain_sql) {
            Ok(s) => s,
            Err(e) => return format!("SQL error: {e}"),
        };

        let rows_result: rusqlite::Result<Vec<(i32, String)>> = stmt
            .query_map(rusqlite::params_from_iter(json_params.iter()), |row| {
                // EXPLAIN QUERY PLAN: (id, parent, notused, detail)
                let id: i32 = row.get(0)?;
                let detail: String = row.get(3)?;
                Ok((id, detail))
            })
            .and_then(|mapped| mapped.collect());

        match rows_result {
            Err(e) => format!("SQL error: {e}"),
            Ok(rows) if rows.is_empty() => "No query plan available.".to_string(),
            Ok(rows) => {
                let mut lines = vec!["Query plan:".to_string(), String::new()];
                for (id, detail) in &rows {
                    let indent = "  ".repeat(*id as usize);
                    lines.push(format!("{indent}{detail}"));
                }
                lines.join("\n")
            }
        }
    }

    #[tool(
        description = "Insert many rows into a pre-existing table in one transaction. 50-100x faster than repeated execute_sql INSERTs. Table must already exist — create it with execute_sql first if needed. conflict: 'error' (default, rolls back batch on violation), 'replace' (upsert), 'ignore' (skip violating rows silently)."
    )]
    async fn bulk_insert(&self, params: Parameters<BulkInsertParams>) -> String {
        let BulkInsertParams {
            alias,
            table,
            columns,
            rows,
            conflict,
        } = params.0;
        let conflict_kw = match conflict.as_deref().unwrap_or("error") {
            "replace" => "OR REPLACE ",
            "ignore" => "OR IGNORE ",
            _ => "",
        };

        if rows.is_empty() {
            return "No rows to insert.".to_string();
        }
        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }
        for col in &columns {
            if let Err(e) = validate_identifier(col, "column name") {
                return e;
            }
        }

        let col_clause: String = columns
            .iter()
            .map(|c| format!("\"{}\"", escape_identifier(c)))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders: String = (0..columns.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT {conflict_kw}INTO \"{}\" ({col_clause}) VALUES ({placeholders})",
            escape_identifier(&table)
        );

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let result: rusqlite::Result<()> = (|| {
            db.conn.execute_batch("BEGIN TRANSACTION")?;
            let mut stmt = db.conn.prepare(&sql)?;
            for row in &rows {
                let json_params: Vec<JsonParam> = row.iter().cloned().map(JsonParam).collect();
                stmt.execute(rusqlite::params_from_iter(json_params.iter()))?;
            }
            drop(stmt);
            db.conn.execute_batch("COMMIT")?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                info!("Bulk-inserted {} rows into '{alias}.{table}'", rows.len());
                format!("Inserted {} rows into '{table}'.", rows.len())
            }
            Err(e) => {
                let _ = db.conn.execute_batch("ROLLBACK");
                format!("Bulk insert error (rolled back): {e}")
            }
        }
    }

    #[tool(
        description = "Import inline CSV text into a table. Table auto-created from the header row with inferred types if absent. if_exists: 'error' (default), 'replace' (drop+recreate), 'append'. For large datasets, prefer import_csv_file to avoid oversized arguments."
    )]
    async fn import_csv(&self, params: Parameters<ImportCsvParams>) -> String {
        let ImportCsvParams {
            alias,
            csv_text,
            table,
            if_exists,
        } = params.0;
        let table = table.unwrap_or_else(|| "imported".to_string());
        let if_exists = if_exists.unwrap_or_else(|| "error".to_string());

        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }

        let mut rdr = csv_crate::Reader::from_reader(io::Cursor::new(csv_text.as_bytes()));
        let headers: Vec<String> = match rdr.headers() {
            Ok(h) => h.iter().map(|s| s.trim().to_string()).collect(),
            Err(e) => return format!("CSV parse error: {e}"),
        };

        if headers.is_empty() {
            return "CSV header row is empty.".to_string();
        }
        for h in &headers {
            if let Err(e) = validate_identifier(h, "column name (from CSV header)") {
                return e;
            }
        }

        let rows: Vec<Vec<String>> = rdr
            .records()
            .filter_map(|r| r.ok())
            .filter(|r| !r.is_empty())
            .map(|r| r.iter().map(|s| s.to_string()).collect())
            .collect();

        if rows.is_empty() {
            return "CSV has a header row but no data rows.".to_string();
        }

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let escaped_table = escape_identifier(&table);
        let col_types: Vec<&'static str> = (0..headers.len())
            .map(|ci| {
                let samples: Vec<&str> = rows
                    .iter()
                    .take(200)
                    .filter_map(|r| r.get(ci).map(|s| s.as_str()))
                    .collect();
                infer_sqlite_type(&samples)
            })
            .collect();
        let col_defs: String = headers
            .iter()
            .zip(col_types.iter())
            .map(|(h, t)| format!("\"{}\" {t}", escape_identifier(h)))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders: String = (0..headers.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        let insert_sql = format!("INSERT INTO \"{escaped_table}\" VALUES ({placeholders})");

        let result: rusqlite::Result<()> = (|| {
            if if_exists == "replace" {
                db.conn
                    .execute_batch(&format!("DROP TABLE IF EXISTS \"{escaped_table}\""))?;
            } else if if_exists == "error" {
                let exists: bool = db.conn.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [&table],
                    |row| row.get::<_, i64>(0),
                )? > 0;
                if exists {
                    return Err(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                        Some(format!(
                            "Table '{table}' already exists. Use if_exists='replace' to overwrite or if_exists='append' to add rows."
                        )),
                    ));
                }
            }
            db.conn.execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS \"{escaped_table}\" ({col_defs})"
            ))?;
            db.conn.execute_batch("BEGIN TRANSACTION")?;
            let mut stmt = db.conn.prepare(&insert_sql)?;
            for row in &rows {
                let row_refs: Vec<&dyn rusqlite::ToSql> =
                    row.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
                stmt.execute(row_refs.as_slice())?;
            }
            drop(stmt);
            db.conn.execute_batch("COMMIT")?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                info!("Imported {} rows into '{alias}.{table}'", rows.len());
                format!(
                    "Imported {} rows into '{table}' ({} columns: {}).",
                    rows.len(),
                    headers.len(),
                    headers.join(", ")
                )
            }
            Err(e) => {
                let _ = db.conn.execute_batch("ROLLBACK");
                format!("Import error: {e}")
            }
        }
    }

    #[tool(
        description = "Run a SELECT query and return results as CSV text. Paginate with offset+max_rows — increment offset by rows returned when the response indicates more are available."
    )]
    async fn export_query_csv(&self, params: Parameters<ExportQueryCsvParams>) -> String {
        let ExportQueryCsvParams {
            alias,
            sql,
            params: bind_params,
            max_rows,
            offset,
        } = params.0;
        let max_rows = max_rows.unwrap_or(self.default_export_max_rows);
        let offset = offset.unwrap_or(0);
        let bind_params = bind_params.unwrap_or_default();
        let json_params: Vec<JsonParam> = bind_params.into_iter().map(JsonParam).collect();

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let mut stmt = match db.conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => return format!("SQL error: {e}"),
        };

        if stmt.column_count() == 0 {
            return "Statement did not return results.".to_string();
        }

        let col_names: Vec<String> = stmt.column_names().iter().map(|&s| s.to_string()).collect();
        let col_count = stmt.column_count();

        let mut output = Vec::new();
        let mut wtr = csv_crate::Writer::from_writer(&mut output);
        if let Err(e) = wtr.write_record(&col_names) {
            return format!("CSV error: {e}");
        }

        let rows_result: rusqlite::Result<(usize, bool)> = stmt
            .query_map(rusqlite::params_from_iter(json_params.iter()), |row| {
                let vals: Vec<String> = (0..col_count)
                    .map(|i| row.get_ref(i).map(value_ref_to_string).unwrap_or_default())
                    .collect();
                Ok(vals)
            })
            .and_then(|mapped| {
                let mut count = 0usize;
                let mut skipped = 0usize;
                let mut truncated = false;
                for row in mapped {
                    let r = row?;
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }
                    if max_rows > 0 && count >= max_rows {
                        truncated = true;
                        break;
                    }
                    wtr.write_record(&r)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                    count += 1;
                }
                Ok((count, truncated))
            });

        if let Err(e) = wtr.flush() {
            return format!("CSV write error: {e}");
        }
        drop(wtr);

        match rows_result {
            Err(e) => format!("SQL error: {e}"),
            Ok((count, truncated)) => {
                let mut csv_text = String::from_utf8_lossy(&output).into_owned();
                if truncated {
                    let start = offset + 1;
                    let end = offset + count;
                    csv_text.push_str(&format!(
                        "\n# Rows {start}-{end}, more available. Call again with offset={end}.\n"
                    ));
                } else if offset > 0 {
                    let start = offset + 1;
                    let end = offset + count;
                    csv_text.push_str(&format!("\n# Rows {start}-{end}, end of results.\n"));
                }
                csv_text
            }
        }
    }

    #[tool(
        description = "Inspect a CSV or TSV file on disk: detect the delimiter, identify column names from the header row, sample data rows, and infer likely SQLite column types. Use this before import_csv_file to understand an unfamiliar file's structure."
    )]
    async fn inspect_csv_file(&self, params: Parameters<InspectCsvFileParams>) -> String {
        let InspectCsvFileParams {
            path,
            delimiter,
            sample_rows,
        } = params.0;
        let sample_rows = sample_rows.unwrap_or(5);

        let file_path = match self.resolve_file_path(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let raw = match std::fs::read(&file_path) {
            Ok(b) => b,
            Err(e) => return format!("Cannot read file '{}': {e}", file_path.display()),
        };

        let delim_byte = match parse_delimiter(&delimiter, &file_path) {
            Ok(b) => b,
            Err(e) => return e,
        };

        let mut rdr = csv_crate::ReaderBuilder::new()
            .delimiter(delim_byte)
            .has_headers(true)
            .flexible(true)
            .from_reader(std::io::Cursor::new(&raw));

        let headers: Vec<String> = match rdr.headers() {
            Ok(h) => h.iter().map(|s| s.trim().to_string()).collect(),
            Err(e) => return format!("CSV parse error: {e}"),
        };

        let mut samples: Vec<Vec<String>> = Vec::new();
        for result in rdr.records().take(sample_rows) {
            match result {
                Ok(r) => samples.push(r.iter().map(|s| s.to_string()).collect()),
                Err(_) => break,
            }
        }

        let delim_name = if delim_byte == b'\t' {
            "TSV (tab-separated)"
        } else {
            "CSV (comma-separated)"
        };
        let mut out = format!(
            "File: {}\nFormat: {}\nColumns: {}\nSample rows shown: {}\n\nColumn schema (inferred):\n",
            file_path.display(),
            delim_name,
            headers.len(),
            samples.len(),
        );
        for (i, col) in headers.iter().enumerate() {
            let col_vals: Vec<&str> = samples
                .iter()
                .filter_map(|r| r.get(i).map(|s| s.as_str()))
                .collect();
            let inferred = infer_sqlite_type(&col_vals);
            out.push_str(&format!("  {col}  →  {inferred}\n"));
        }
        if !samples.is_empty() {
            out.push_str(&format!("\nSample data ({} row(s)):\n", samples.len()));
            out.push_str(&headers.join("\t"));
            out.push('\n');
            for row in &samples {
                out.push_str(&row.join("\t"));
                out.push('\n');
            }
        }
        out
    }

    #[tool(
        description = "Import a CSV or TSV file from disk into a table. Delimiter auto-detected (.tsv→tab, others→comma). Table auto-created with inferred types if absent. IMPORTANT: if_exists defaults to 'error' — pass 'replace' to overwrite or 'append' to extend. Supply columns when has_headers=false."
    )]
    async fn import_csv_file(&self, params: Parameters<ImportCsvFileParams>) -> String {
        let ImportCsvFileParams {
            alias,
            path,
            table,
            has_headers,
            columns,
            delimiter,
            if_exists,
        } = params.0;
        let has_headers = has_headers.unwrap_or(true);
        let if_exists = if_exists.unwrap_or_else(|| "error".to_string());

        let file_path = match self.resolve_file_path(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let delim_byte = match parse_delimiter(&delimiter, &file_path) {
            Ok(b) => b,
            Err(e) => return e,
        };

        let raw = match std::fs::read(&file_path) {
            Ok(b) => b,
            Err(e) => return format!("Cannot read file '{}': {e}", file_path.display()),
        };

        let mut rdr = csv_crate::ReaderBuilder::new()
            .delimiter(delim_byte)
            .has_headers(has_headers)
            .flexible(true)
            .from_reader(std::io::Cursor::new(&raw));

        let col_names: Vec<String> = if has_headers {
            match rdr.headers() {
                Ok(h) => h.iter().map(|s| s.trim().to_string()).collect(),
                Err(e) => return format!("CSV parse error: {e}"),
            }
        } else {
            match columns {
                Some(c) => c,
                None => return "has_headers is false: you must supply 'columns' with the column names to use.".to_string(),
            }
        };

        if col_names.is_empty() {
            return "No columns found.".to_string();
        }
        for c in &col_names {
            if let Err(e) = validate_identifier(c, "column name") {
                return e;
            }
        }

        let table_name = match table {
            Some(t) => t,
            None => file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("imported")
                .to_string(),
        };
        if let Err(e) = validate_identifier(&table_name, "table name") {
            return e;
        }

        let rows: Vec<Vec<String>> = rdr
            .records()
            .filter_map(|r| r.ok())
            .filter(|r| !r.is_empty())
            .map(|r| r.iter().map(|s| s.to_string()).collect())
            .collect();

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let escaped_table = escape_identifier(&table_name);
        // Infer per-column types from sampled row data
        let col_types: Vec<&'static str> = (0..col_names.len())
            .map(|ci| {
                let samples: Vec<&str> = rows
                    .iter()
                    .take(200)
                    .filter_map(|r| r.get(ci).map(|s| s.as_str()))
                    .collect();
                infer_sqlite_type(&samples)
            })
            .collect();
        let col_defs: String = col_names
            .iter()
            .zip(col_types.iter())
            .map(|(c, t)| format!("\"{}\" {t}", escape_identifier(c)))
            .collect::<Vec<_>>()
            .join(", ");
        let placeholders: String = (0..col_names.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");
        let insert_sql = format!("INSERT INTO \"{escaped_table}\" VALUES ({placeholders})");

        let result: rusqlite::Result<()> = (|| {
            if if_exists == "replace" {
                db.conn
                    .execute_batch(&format!("DROP TABLE IF EXISTS \"{escaped_table}\""))?;
            } else if if_exists == "error" {
                let exists: bool = db.conn.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [&table_name],
                    |row| row.get::<_, i64>(0),
                )? > 0;
                if exists {
                    return Err(rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                        Some(format!(
                            "Table '{table_name}' already exists. Use if_exists='replace' to overwrite or if_exists='append' to add rows."
                        )),
                    ));
                }
            }
            db.conn.execute_batch(&format!(
                "CREATE TABLE IF NOT EXISTS \"{escaped_table}\" ({col_defs})"
            ))?;
            db.conn.execute_batch("BEGIN TRANSACTION")?;
            let mut stmt = db.conn.prepare(&insert_sql)?;
            for row in &rows {
                let row_refs: Vec<&dyn rusqlite::ToSql> =
                    row.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
                stmt.execute(row_refs.as_slice())?;
            }
            drop(stmt);
            db.conn.execute_batch("COMMIT")?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                info!(
                    "Imported {} rows from '{}' into '{alias}.{table_name}'",
                    rows.len(),
                    file_path.display()
                );
                format!(
                    "Imported {} rows into '{table_name}' from '{}' ({} columns: {}).",
                    rows.len(),
                    file_path.display(),
                    col_names.len(),
                    col_names.join(", ")
                )
            }
            Err(e) => {
                let _ = db.conn.execute_batch("ROLLBACK");
                format!("Import error: {e}")
            }
        }
    }

    #[tool(
        description = "Run a SELECT query and write results to a CSV or TSV file on disk. Use for large exports that would be unwieldy as inline text. Supports optional header row and custom delimiter. Returns a row count summary."
    )]
    async fn export_query_csv_file(&self, params: Parameters<ExportQueryCsvFileParams>) -> String {
        let ExportQueryCsvFileParams {
            alias,
            sql,
            params: bind_params,
            path,
            emit_headers,
            delimiter,
            max_rows,
        } = params.0;
        let emit_headers = emit_headers.unwrap_or(true);
        let max_rows = max_rows.unwrap_or(self.default_export_max_rows);
        let bind_params = bind_params.unwrap_or_default();
        let json_params: Vec<JsonParam> = bind_params.into_iter().map(JsonParam).collect();

        let file_path = match self.resolve_file_path(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let delim_byte = match parse_delimiter(&delimiter, &file_path) {
            Ok(b) => b,
            Err(e) => return e,
        };

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let mut stmt = match db.conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => return format!("SQL error: {e}"),
        };

        if stmt.column_count() == 0 {
            return "Statement did not return results.".to_string();
        }

        let col_names: Vec<String> = stmt.column_names().iter().map(|&s| s.to_string()).collect();
        let col_count = stmt.column_count();

        let file = match std::fs::File::create(&file_path) {
            Ok(f) => f,
            Err(e) => return format!("Cannot create file '{}': {e}", file_path.display()),
        };

        let mut wtr = csv_crate::WriterBuilder::new()
            .delimiter(delim_byte)
            .from_writer(file);

        if emit_headers {
            if let Err(e) = wtr.write_record(&col_names) {
                return format!("Write error: {e}");
            }
        }

        let rows_result: rusqlite::Result<(usize, bool)> = stmt
            .query_map(rusqlite::params_from_iter(json_params.iter()), |row| {
                let vals: Vec<String> = (0..col_count)
                    .map(|i| row.get_ref(i).map(value_ref_to_string).unwrap_or_default())
                    .collect();
                Ok(vals)
            })
            .and_then(|mapped| {
                let mut count = 0usize;
                let mut truncated = false;
                for row in mapped {
                    let r = row?;
                    if max_rows > 0 && count >= max_rows {
                        truncated = true;
                        break;
                    }
                    wtr.write_record(&r)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                    count += 1;
                }
                Ok((count, truncated))
            });

        if let Err(e) = wtr.flush() {
            return format!("Write error: {e}");
        }
        drop(wtr);

        match rows_result {
            Err(e) => format!("SQL error: {e}"),
            Ok((count, truncated)) => {
                info!("Exported {count} rows to '{}'", file_path.display());
                let mut msg = format!(
                    "Exported {count} rows to '{}' ({} columns).",
                    file_path.display(),
                    col_names.len()
                );
                if truncated {
                    msg.push_str(&format!(" Truncated at {max_rows} rows."));
                }
                msg
            }
        }
    }

    #[tool(
        description = "Import an inline JSON array of objects into a table. Table auto-created with types inferred from JSON values if absent. if_exists: 'error' (default), 'replace', 'append'. For large JSON, use import_json_file instead."
    )]
    async fn import_json(&self, params: Parameters<ImportJsonParams>) -> String {
        let ImportJsonParams {
            alias,
            json_text,
            table,
            if_exists,
        } = params.0;
        let table = table.unwrap_or_else(|| "imported".to_string());
        let if_exists = if_exists.unwrap_or_else(|| "error".to_string());

        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }

        let arr: Vec<serde_json::Map<String, serde_json::Value>> =
            match serde_json::from_str::<serde_json::Value>(&json_text) {
                Ok(serde_json::Value::Array(arr)) => arr
                    .into_iter()
                    .filter_map(|v| {
                        if let serde_json::Value::Object(m) = v {
                            Some(m)
                        } else {
                            None
                        }
                    })
                    .collect(),
                Ok(_) => return "JSON must be an array of objects.".to_string(),
                Err(e) => return format!("JSON parse error: {e}"),
            };

        if arr.is_empty() {
            return "JSON array is empty.".to_string();
        }

        // Collect column names from the first object
        let col_names: Vec<String> = arr[0].keys().cloned().collect();
        for c in &col_names {
            if let Err(e) = validate_identifier(c, "column name") {
                return e;
            }
        }

        import_json_objects(
            &self.connections,
            &alias,
            &table,
            &if_exists,
            &col_names,
            &arr,
        )
    }

    #[tool(
        description = "Import a JSON file into a table. format: 'array' (default, JSON array of objects) or 'ndjson' (one object per line, good for large files). if_exists: 'error' (default), 'replace', 'append'. Table auto-created with inferred types if absent."
    )]
    async fn import_json_file(&self, params: Parameters<ImportJsonFileParams>) -> String {
        let ImportJsonFileParams {
            alias,
            path,
            table,
            format,
            if_exists,
        } = params.0;
        let if_exists = if_exists.unwrap_or_else(|| "error".to_string());
        let format = format.unwrap_or_else(|| "array".to_string());

        let file_path = match self.resolve_file_path(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let table = match table {
            Some(t) => t,
            None => file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("imported")
                .to_string(),
        };
        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }

        let raw = match std::fs::read_to_string(&file_path) {
            Ok(s) => s,
            Err(e) => return format!("Cannot read file '{}': {e}", file_path.display()),
        };

        let arr: Vec<serde_json::Map<String, serde_json::Value>> = if format == "ndjson" {
            let mut objs = Vec::new();
            for (i, line) in raw.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(serde_json::Value::Object(m)) => objs.push(m),
                    Ok(_) => return format!("Line {}: expected a JSON object.", i + 1),
                    Err(e) => return format!("Line {}: JSON parse error: {e}", i + 1),
                }
            }
            objs
        } else {
            match serde_json::from_str::<serde_json::Value>(&raw) {
                Ok(serde_json::Value::Array(arr)) => arr
                    .into_iter()
                    .filter_map(|v| {
                        if let serde_json::Value::Object(m) = v {
                            Some(m)
                        } else {
                            None
                        }
                    })
                    .collect(),
                Ok(_) => {
                    return "JSON file must contain an array of objects (or use format='ndjson')."
                        .to_string();
                }
                Err(e) => return format!("JSON parse error: {e}"),
            }
        };

        if arr.is_empty() {
            return "JSON source is empty.".to_string();
        }

        let col_names: Vec<String> = arr[0].keys().cloned().collect();
        for c in &col_names {
            if let Err(e) = validate_identifier(c, "column name") {
                return e;
            }
        }

        let result = import_json_objects(
            &self.connections,
            &alias,
            &table,
            &if_exists,
            &col_names,
            &arr,
        );
        info!("Imported from JSON file '{}'", file_path.display());
        result
    }

    #[tool(
        description = "Run a SELECT query and write results to a JSON file: either a JSON array of objects (format='array', default) or NDJSON with one object per line (format='ndjson', streaming-friendly for large files)."
    )]
    async fn export_query_json_file(
        &self,
        params: Parameters<ExportQueryJsonFileParams>,
    ) -> String {
        let ExportQueryJsonFileParams {
            alias,
            sql,
            params: bind_params,
            path,
            format,
            max_rows,
        } = params.0;
        let format = format.unwrap_or_else(|| "array".to_string());
        let max_rows = max_rows.unwrap_or(self.default_export_max_rows);
        let bind_params = bind_params.unwrap_or_default();
        let json_params: Vec<JsonParam> = bind_params.into_iter().map(JsonParam).collect();

        let file_path = match self.resolve_file_path(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let mut stmt = match db.conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => return format!("SQL error: {e}"),
        };
        if stmt.column_count() == 0 {
            return "Statement did not return results.".to_string();
        }

        let col_names: Vec<String> = stmt.column_names().iter().map(|&s| s.to_string()).collect();
        let col_count = stmt.column_count();

        let param_refs: Vec<&dyn rusqlite::ToSql> = json_params
            .iter()
            .map(|p| p as &dyn rusqlite::ToSql)
            .collect();
        let rows_iter = match stmt.query(param_refs.as_slice()) {
            Ok(r) => r,
            Err(e) => return format!("Query error: {e}"),
        };

        // Collect rows as JSON objects
        let mut rows: Vec<serde_json::Map<String, serde_json::Value>> = Vec::new();
        let mut truncated = false;
        let mut rows_iter = rows_iter;
        loop {
            match rows_iter.next() {
                Err(e) => return format!("Row error: {e}"),
                Ok(None) => break,
                Ok(Some(row)) => {
                    if max_rows > 0 && rows.len() >= max_rows {
                        truncated = true;
                        break;
                    }
                    let mut obj = serde_json::Map::new();
                    for i in 0..col_count {
                        let val = match row.get_ref(i) {
                            Ok(ValueRef::Null) => serde_json::Value::Null,
                            Ok(ValueRef::Integer(n)) => serde_json::Value::Number(n.into()),
                            Ok(ValueRef::Real(f)) => serde_json::Number::from_f64(f)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null),
                            Ok(ValueRef::Text(t)) => {
                                serde_json::Value::String(String::from_utf8_lossy(t).into_owned())
                            }
                            Ok(ValueRef::Blob(b)) => {
                                serde_json::Value::String(format!("<blob {} bytes>", b.len()))
                            }
                            Err(_) => serde_json::Value::Null,
                        };
                        obj.insert(col_names[i].clone(), val);
                    }
                    rows.push(obj);
                }
            }
        }
        drop(rows_iter);
        drop(stmt);

        // Write to file
        let write_result: std::io::Result<()> = (|| {
            use std::io::Write;
            let mut f = std::fs::File::create(&file_path)?;
            if format == "ndjson" {
                for obj in &rows {
                    let line = serde_json::to_string(obj).unwrap();
                    f.write_all(line.as_bytes())?;
                    f.write_all(b"\n")?;
                }
            } else {
                let json = serde_json::to_string_pretty(&rows).unwrap();
                f.write_all(json.as_bytes())?;
                f.write_all(b"\n")?;
            }
            Ok(())
        })();

        match write_result {
            Err(e) => format!("Write error: {e}"),
            Ok(()) => {
                info!(
                    "Exported {} rows to '{}' as JSON {format}",
                    rows.len(),
                    file_path.display()
                );
                let mut msg = format!(
                    "Exported {} rows to '{}' ({format}, {} columns).",
                    rows.len(),
                    file_path.display(),
                    col_names.len()
                );
                if truncated {
                    msg.push_str(&format!(" Truncated at {max_rows} rows."));
                }
                msg
            }
        }
    }

    #[tool(
        description = "Run a SELECT query and return results as a Markdown table for display in chat. Pipe characters in cells are auto-escaped. Paginate with offset+max_rows the same way as execute_sql."
    )]
    async fn export_query_markdown(&self, params: Parameters<ExportQueryMarkdownParams>) -> String {
        let ExportQueryMarkdownParams {
            alias,
            sql,
            params: bind_params,
            max_rows,
            offset,
        } = params.0;
        let max_rows = max_rows.unwrap_or(self.default_max_rows);
        let offset = offset.unwrap_or(0);
        let bind_params = bind_params.unwrap_or_default();
        let json_params: Vec<JsonParam> = bind_params.into_iter().map(JsonParam).collect();

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let mut stmt = match db.conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => return format!("SQL error: {e}"),
        };
        if stmt.column_count() == 0 {
            return "Statement did not return results.".to_string();
        }

        let col_names: Vec<String> = stmt.column_names().iter().map(|&s| s.to_string()).collect();
        let col_count = stmt.column_count();

        let param_refs: Vec<&dyn rusqlite::ToSql> = json_params
            .iter()
            .map(|p| p as &dyn rusqlite::ToSql)
            .collect();
        let rows_iter = match stmt.query(param_refs.as_slice()) {
            Ok(r) => r,
            Err(e) => return format!("Query error: {e}"),
        };

        let mut data: Vec<Vec<String>> = Vec::new();
        let mut truncated = false;
        let mut skipped = 0usize;
        let mut rows_iter = rows_iter;
        loop {
            match rows_iter.next() {
                Err(e) => return format!("Row error: {e}"),
                Ok(None) => break,
                Ok(Some(row)) => {
                    if skipped < offset {
                        skipped += 1;
                        continue;
                    }
                    if max_rows > 0 && data.len() >= max_rows {
                        truncated = true;
                        break;
                    }
                    let cells: Vec<String> = (0..col_count)
                        .map(|i| value_ref_to_string(row.get_ref(i).unwrap_or(ValueRef::Null)))
                        .map(|s| s.replace('|', "\\|").replace('\n', " "))
                        .collect();
                    data.push(cells);
                }
            }
        }
        drop(rows_iter);
        drop(stmt);

        if data.is_empty() {
            return if offset > 0 {
                format!("(No rows at offset {offset} -- past end of results.)")
            } else {
                "(no rows)".to_string()
            };
        }

        // Calculate column widths
        let mut widths: Vec<usize> = col_names.iter().map(|h| h.len()).collect();
        for row in &data {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        // Render
        let mut out = String::new();
        // Header
        out.push('|');
        for (i, h) in col_names.iter().enumerate() {
            out.push_str(&format!(" {:width$} |", h, width = widths[i]));
        }
        out.push('\n');
        // Separator
        out.push('|');
        for w in &widths {
            out.push_str(&format!(" {} |", "-".repeat(*w)));
        }
        out.push('\n');
        // Rows
        for row in &data {
            out.push('|');
            for (i, cell) in row.iter().enumerate() {
                let w = widths.get(i).copied().unwrap_or(0);
                out.push_str(&format!(" {:width$} |", cell, width = w));
            }
            out.push('\n');
        }

        if truncated {
            let start = offset + 1;
            let end = offset + data.len();
            out.push_str(&format!(
                "\n_(Rows {start}-{end}, more available. Call again with `offset={end}`.)_\n"
            ));
        } else if offset > 0 {
            let start = offset + 1;
            let end = offset + data.len();
            out.push_str(&format!("\n_(Rows {start}-{end}, end of results.)_\n"));
        }
        out
    }

    #[tool(
        description = "Profile a table: row count, NULL count, distinct count, min/max, and avg per column. A one-call replacement for many exploratory SELECT queries on an unfamiliar dataset."
    )]
    async fn table_stats(&self, params: Parameters<TableStatsParams>) -> String {
        let TableStatsParams {
            alias,
            table,
            columns,
        } = params.0;
        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let safe_table = escape_identifier(&table);

        let all_cols: Vec<(String, String)> = {
            let mut stmt = db
                .conn
                .prepare(&format!("PRAGMA table_info(\"{safe_table}\")"))
                .ok();
            stmt.as_mut()
                .and_then(|s| {
                    s.query_map([], |row| {
                        Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
                    })
                    .ok()
                    .map(|m| m.flatten().collect())
                })
                .unwrap_or_default()
        };

        if all_cols.is_empty() {
            return format!("Table '{table}' not found.");
        }

        let col_info: Vec<(String, String)> = if let Some(requested) = columns {
            all_cols
                .into_iter()
                .filter(|(n, _)| requested.contains(n))
                .collect()
        } else {
            all_cols
        };

        let total_rows: i64 = match db.conn.query_row(
            &format!("SELECT COUNT(*) FROM \"{safe_table}\""),
            [],
            |row| row.get(0),
        ) {
            Ok(n) => n,
            Err(e) => return format!("Error counting rows: {e}"),
        };

        let mut lines = vec![
            format!("Table: {table}"),
            format!("Total rows: {total_rows}"),
        ];

        for (col_name, col_type) in &col_info {
            let safe_col = escape_identifier(col_name);
            let stats: Option<(i64, i64, Option<String>, Option<String>, Option<f64>)> = db
                .conn
                .query_row(
                    &format!(
                        "SELECT COUNT(\"{safe_col}\"), COUNT(DISTINCT \"{safe_col}\"), \
                         CAST(MIN(\"{safe_col}\") AS TEXT), CAST(MAX(\"{safe_col}\") AS TEXT), \
                         AVG(CAST(\"{safe_col}\" AS REAL)) FROM \"{safe_table}\""
                    ),
                    [],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<f64>>(4)?,
                        ))
                    },
                )
                .ok();

            lines.push(String::new());
            if let Some((non_null, distinct, min_val, max_val, avg_val)) = stats {
                let null_count = total_rows - non_null;
                let type_str = if col_type.is_empty() {
                    "?".to_string()
                } else {
                    col_type.clone()
                };
                let null_str = if null_count == 0 {
                    "0 nulls".to_string()
                } else {
                    format!(
                        "{null_count} nulls ({:.1}%)",
                        null_count as f64 / total_rows as f64 * 100.0
                    )
                };
                let min_str = min_val.as_deref().unwrap_or("NULL");
                let max_str = max_val.as_deref().unwrap_or("NULL");
                let min_d = if min_str.len() > 30 {
                    format!("{}...", &min_str[..30])
                } else {
                    min_str.to_string()
                };
                let max_d = if max_str.len() > 30 {
                    format!("{}...", &max_str[..30])
                } else {
                    max_str.to_string()
                };
                lines.push(format!("  {col_name}  [{type_str}]  non_null: {non_null}  {null_str}  distinct: {distinct}"));
                lines.push(format!("    min: {min_d}  max: {max_d}"));
                if let Some(avg) = avg_val {
                    lines.push(format!("    avg: {avg:.4}"));
                }
            } else {
                lines.push(format!("  {col_name}  (error reading stats)"));
            }
        }

        lines.join("\n")
    }

    #[tool(
        description = "Attach a second SQLite file to an open connection. Its tables become accessible as schema_name.table_name for cross-database JOINs and copies. No new alias is created — all queries run on the main alias. Call detach_database when done to release the file lock."
    )]
    async fn attach_database(&self, params: Parameters<AttachDatabaseParams>) -> String {
        let AttachDatabaseParams {
            alias,
            path,
            schema_name,
        } = params.0;
        if let Err(e) = validate_identifier(&schema_name, "schema name") {
            return e;
        }
        let file_path = match self.resolve_db_path(&path) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };
        let path_str = file_path.to_string_lossy().replace('\'', "''");
        let safe_schema = escape_identifier(&schema_name);
        let sql = format!("ATTACH DATABASE '{path_str}' AS \"{safe_schema}\"");
        match db.conn.execute_batch(&sql) {
            Ok(()) => {
                info!(
                    "Attached '{}' as schema '{schema_name}' on '{alias}'",
                    file_path.display()
                );
                format!(
                    "Attached '{}' as schema '{schema_name}'. \
                     Tables are accessible as \"{schema_name}\".table_name in queries.",
                    file_path.display()
                )
            }
            Err(e) => format!("ATTACH error: {e}"),
        }
    }

    #[tool(
        description = "Remove a previously attached schema from a connection and release its file lock. Does not close the main connection or delete any files."
    )]
    async fn detach_database(&self, params: Parameters<DetachDatabaseParams>) -> String {
        let DetachDatabaseParams { alias, schema_name } = params.0;
        if let Err(e) = validate_identifier(&schema_name, "schema name") {
            return e;
        }
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };
        let safe_schema = escape_identifier(&schema_name);
        match db
            .conn
            .execute_batch(&format!("DETACH DATABASE \"{safe_schema}\""))
        {
            Ok(()) => {
                info!("Detached schema '{schema_name}' from '{alias}'");
                format!("Detached schema '{schema_name}'.")
            }
            Err(e) => format!("DETACH error: {e}"),
        }
    }

    #[tool(
        description = "Create an index to speed up WHERE, JOIN, and ORDER BY on the indexed columns. Name auto-generated as idx_table_col1_col2 if omitted. unique=true enforces uniqueness. where_clause creates a partial (filtered) index. Use explain_query to verify index usage."
    )]
    async fn create_index(&self, params: Parameters<CreateIndexParams>) -> String {
        let CreateIndexParams {
            alias,
            table,
            columns,
            index_name,
            unique,
            where_clause,
            if_not_exists,
        } = params.0;
        let unique = unique.unwrap_or(false);
        let if_not_exists = if_not_exists.unwrap_or(false);

        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }
        if columns.is_empty() {
            return "At least one column is required.".to_string();
        }
        for col in &columns {
            if let Err(e) = validate_identifier(col, "column name") {
                return e;
            }
        }

        let idx_name = match index_name {
            Some(n) => n,
            None => format!("idx_{}_{}", table, columns.join("_")),
        };
        if let Err(e) = validate_identifier(&idx_name, "index name") {
            return e;
        }

        let unique_kw = if unique { "UNIQUE " } else { "" };
        let ine_kw = if if_not_exists { "IF NOT EXISTS " } else { "" };
        let col_list = columns
            .iter()
            .map(|c| format!("\"{}\"", escape_identifier(c)))
            .collect::<Vec<_>>()
            .join(", ");
        let where_part = where_clause
            .as_deref()
            .map(|w| format!(" WHERE {w}"))
            .unwrap_or_default();
        let sql = format!(
            "CREATE {unique_kw}INDEX {ine_kw}\"{}\" ON \"{}\" ({col_list}){where_part}",
            escape_identifier(&idx_name),
            escape_identifier(&table),
        );

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };
        match db.conn.execute_batch(&sql) {
            Ok(()) => {
                info!("Created index '{idx_name}' on '{alias}.{table}'");
                let kind = if unique { "UNIQUE INDEX" } else { "INDEX" };
                format!(
                    "Created {kind} '{idx_name}' on '{table}' ({}).{where_part}",
                    columns.join(", ")
                )
            }
            Err(e) => format!("CREATE INDEX error: {e}"),
        }
    }

    #[tool(
        description = "Drop an index. Never deletes data — only removes the index structure. Call without confirm to preview (table name, DDL). Call with confirm=true to execute."
    )]
    async fn drop_index(&self, params: Parameters<DropIndexParams>) -> String {
        let DropIndexParams {
            alias,
            index_name,
            confirm,
        } = params.0;
        let confirm = confirm.unwrap_or(false);

        if let Err(e) = validate_identifier(&index_name, "index name") {
            return e;
        }

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let info: Option<(String, Option<String>)> = db
            .conn
            .query_row(
                "SELECT tbl_name, sql FROM sqlite_master WHERE type='index' AND name=?1",
                rusqlite::params![index_name],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .ok();

        let (tbl_name, idx_ddl) = match info {
            Some(i) => i,
            None => return format!("Index '{index_name}' not found."),
        };

        if !confirm {
            let ddl_line = idx_ddl
                .as_deref()
                .unwrap_or("(auto-generated — no stored DDL)");
            return format!(
                "Preview: DROP INDEX '{index_name}' on table '{tbl_name}'\nDDL: {ddl_line}\nCall again with confirm=true to drop."
            );
        }

        let safe_idx = escape_identifier(&index_name);
        match db.conn.execute_batch(&format!("DROP INDEX \"{safe_idx}\"")) {
            Ok(()) => {
                info!("Dropped index '{index_name}' from '{alias}.{tbl_name}'");
                format!("Dropped index '{index_name}' (was on table '{tbl_name}').")
            }
            Err(e) => format!("DROP INDEX error: {e}"),
        }
    }

    #[tool(
        description = "Add a column to an existing table (ALTER TABLE ... ADD COLUMN). Defaults to TEXT type. NOT NULL requires a default_value (SQLite restriction)."
    )]
    async fn add_column(&self, params: Parameters<AddColumnParams>) -> String {
        let AddColumnParams {
            alias,
            table,
            column,
            col_type,
            not_null,
            default_value,
        } = params.0;
        let col_type = col_type.unwrap_or_else(|| "TEXT".to_string());
        let not_null = not_null.unwrap_or(false);

        if let Err(e) = validate_identifier(&table, "table name") {
            return e;
        }
        if let Err(e) = validate_identifier(&column, "column name") {
            return e;
        }

        if not_null && default_value.is_none() {
            return "NOT NULL requires a default_value (SQLite requires a non-null default \
                    for columns added via ALTER TABLE ADD COLUMN)."
                .to_string();
        }

        let safe_table = escape_identifier(&table);
        let safe_col = escape_identifier(&column);
        let mut sql = format!("ALTER TABLE \"{safe_table}\" ADD COLUMN \"{safe_col}\" {col_type}");
        if not_null {
            sql.push_str(" NOT NULL");
        }
        if let Some(ref def) = default_value {
            sql.push_str(&format!(" DEFAULT {def}"));
        }

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };
        match db.conn.execute_batch(&sql) {
            Ok(()) => {
                info!("Added column '{column}' ({col_type}) to '{alias}.{table}'");
                let mut desc = format!("Added column '{column}' ({col_type})");
                if not_null {
                    desc.push_str(" NOT NULL");
                }
                if let Some(ref def) = default_value {
                    desc.push_str(&format!(" DEFAULT {def}"));
                }
                desc.push_str(&format!(" to table '{table}'."));
                desc
            }
            Err(e) => format!("ADD COLUMN error: {e}"),
        }
    }

    #[tool(
        description = "List all user tables. Prefer database_info for a full overview including row counts, column schemas, and indexes."
    )]
    async fn list_tables(&self, params: Parameters<AliasParams>) -> String {
        let alias = params.0.alias;
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let result: rusqlite::Result<Vec<String>> = db
            .conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .and_then(|mut stmt| {
                stmt.query_map([], |row| row.get(0))
                    .and_then(|mapped| mapped.collect())
            });

        match result {
            Err(e) => format!("SQL error: {e}"),
            Ok(tables) if tables.is_empty() => "No tables found.".to_string(),
            Ok(tables) => tables.join("\n"),
        }
    }

    #[tool(
        description = "Show a table's column definitions (names, types, constraints, DDL) and all its indexes. Prefer database_info for a multi-table overview."
    )]
    async fn describe_table(&self, params: Parameters<TableParams>) -> String {
        let TableParams { alias, table } = params.0;
        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let safe = escape_identifier(&table);
        let cols_result: rusqlite::Result<Vec<(i32, String, String, i32, Option<String>, i32)>> =
            db.conn
                .prepare(&format!("PRAGMA table_info(\"{safe}\")"))
                .and_then(|mut stmt| {
                    stmt.query_map([], |row| {
                        Ok((
                            row.get::<_, i32>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, i32>(3)?,
                            row.get::<_, Option<String>>(4)?,
                            row.get::<_, i32>(5)?,
                        ))
                    })
                    .and_then(|mapped| mapped.collect())
                });

        let cols = match cols_result {
            Err(e) => return format!("SQL error: {e}"),
            Ok(c) if c.is_empty() => return format!("Table '{table}' not found."),
            Ok(c) => c,
        };

        let mut lines = vec![
            format!("Table: {table}"),
            String::new(),
            format!(
                "{:<4} {:<30} {:<15} {:<10} {:<15} {}",
                "#", "Name", "Type", "Nullable", "Default", "PK"
            ),
            "-".repeat(80),
        ];

        for (cid, name, col_type, notnull, default_val, pk) in &cols {
            let nullable = if *notnull != 0 { "NO" } else { "YES" };
            let default_str = default_val.as_deref().unwrap_or("");
            let pk_str = if *pk != 0 { "YES" } else { "" };
            lines.push(format!(
                "{:<4} {:<30} {:<15} {:<10} {:<15} {}",
                cid, name, col_type, nullable, default_str, pk_str
            ));
        }

        // DDL
        let ddl: Option<String> = db
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table],
                |row| row.get(0),
            )
            .ok()
            .flatten();

        if let Some(ddl) = ddl {
            lines.push(String::new());
            lines.push("DDL:".to_string());
            lines.push(ddl);
        }

        // Indexes
        let safe = escape_identifier(&table);
        let indexes: Vec<(String, i32, String)> = {
            let mut stmt = db
                .conn
                .prepare(&format!("PRAGMA index_list(\"{safe}\")"))
                .ok();
            stmt.as_mut()
                .and_then(|s| {
                    s.query_map([], |row| {
                        Ok((
                            row.get::<_, String>(1)?, // name
                            row.get::<_, i32>(2)?,    // unique
                            row.get::<_, String>(3)?, // origin: c=CREATE INDEX, u=UNIQUE constraint, pk=PRIMARY KEY
                        ))
                    })
                    .ok()
                    .map(|m| m.flatten().collect())
                })
                .unwrap_or_default()
        };

        if !indexes.is_empty() {
            lines.push(String::new());
            lines.push("Indexes:".to_string());
            for (idx_name, unique, origin) in &indexes {
                let safe_idx = escape_identifier(idx_name);
                let cols: Vec<String> = {
                    let mut stmt = db
                        .conn
                        .prepare(&format!("PRAGMA index_info(\"{safe_idx}\")"))
                        .ok();
                    stmt.as_mut()
                        .and_then(|s| {
                            s.query_map([], |row| row.get::<_, String>(2))
                                .ok()
                                .map(|m| m.flatten().collect())
                        })
                        .unwrap_or_default()
                };
                let kind = match origin.as_str() {
                    "pk" => "PRIMARY KEY",
                    "u" => "UNIQUE constraint",
                    _ => {
                        if *unique != 0 {
                            "UNIQUE INDEX"
                        } else {
                            "INDEX"
                        }
                    }
                };
                // Get DDL for explicit CREATE INDEX (origin == "c")
                let idx_ddl: Option<String> = if origin == "c" {
                    db.conn
                        .query_row(
                            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?1",
                            rusqlite::params![idx_name],
                            |row| row.get(0),
                        )
                        .ok()
                        .flatten()
                } else {
                    None
                };
                lines.push(format!(
                    "  {idx_name}  [{kind}]  columns: ({})",
                    cols.join(", ")
                ));
                if let Some(ddl) = idx_ddl {
                    lines.push(format!("    {ddl}"));
                }
            }
        }

        lines.join("\n")
    }

    #[tool(
        description = "Drop a table permanently. Call without confirm to preview (row count, columns). Call with confirm=true to execute. Irreversible — verify you no longer need the data before confirming."
    )]
    async fn drop_table(&self, params: Parameters<DropTableParams>) -> String {
        let DropTableParams {
            alias,
            table,
            confirm,
        } = params.0;
        let confirm = confirm.unwrap_or(false);

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        // Check it exists
        let exists: bool = db
            .conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![table],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return format!("Table '{table}' does not exist.");
        }

        let safe = escape_identifier(&table);

        let count: i64 = db
            .conn
            .query_row(&format!("SELECT COUNT(*) FROM \"{safe}\""), [], |r| {
                r.get(0)
            })
            .unwrap_or(-1);

        let col_names: Vec<String> = db
            .conn
            .prepare(&format!("PRAGMA table_info(\"{safe}\")"))
            .and_then(|mut stmt| {
                stmt.query_map([], |row| row.get::<_, String>(1))
                    .and_then(|m| m.collect())
            })
            .unwrap_or_default();

        if !confirm {
            return format!(
                "Would drop table '{table}' with {count} rows and {} column(s) ({}). \
                 Call again with confirm=true to proceed.",
                col_names.len(),
                col_names.join(", ")
            );
        }

        match db.conn.execute_batch(&format!("DROP TABLE \"{safe}\"")) {
            Ok(()) => {
                info!("Dropped table '{table}' from '{alias}'");
                format!("Dropped table '{table}' ({count} rows deleted).")
            }
            Err(e) => format!("Error dropping table: {e}"),
        }
    }

    #[tool(description = "Rename a table in place. All data, indexes, and triggers are preserved.")]
    async fn rename_table(&self, params: Parameters<RenameTableParams>) -> String {
        let RenameTableParams {
            alias,
            old_name,
            new_name,
        } = params.0;

        if let Err(e) = validate_identifier(&new_name, "new table name") {
            return e;
        }

        let conns = self.connections.lock().unwrap();
        let db = match conns.get(&alias) {
            Some(d) => d,
            None => return Self::no_such_alias(&alias, &conns),
        };

        let exists: bool = db
            .conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                rusqlite::params![old_name],
                |_| Ok(true),
            )
            .unwrap_or(false);
        if !exists {
            return format!("Table '{old_name}' does not exist.");
        }

        let old_safe = escape_identifier(&old_name);
        let new_safe = escape_identifier(&new_name);
        match db.conn.execute_batch(&format!(
            "ALTER TABLE \"{old_safe}\" RENAME TO \"{new_safe}\""
        )) {
            Ok(()) => {
                info!("Renamed table '{old_name}' to '{new_name}' in '{alias}'");
                format!("Renamed table '{old_name}' to '{new_name}'.")
            }
            Err(e) => format!("Error renaming table: {e}"),
        }
    }
}

impl LumenSqlite {
    fn no_such_alias(alias: &str, conns: &HashMap<String, OpenDatabase>) -> String {
        let available = if conns.is_empty() {
            "(none)".to_string()
        } else {
            let mut aliases: Vec<&str> = conns.keys().map(|s| s.as_str()).collect();
            aliases.sort();
            aliases.join(", ")
        };
        format!("No open database with alias '{alias}'. Open databases: {available}")
    }
}

// ---------------------------------------------------------------------------
// MCP handler wiring
// ---------------------------------------------------------------------------

#[tool_handler]
impl ServerHandler for LumenSqlite {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(SERVER_INSTRUCTIONS)
            .with_server_info(rmcp::model::Implementation::new(
                "lumen-sqlite",
                env!("CARGO_PKG_VERSION"),
            ))
    }
}

// ---------------------------------------------------------------------------
// Server instructions
// ---------------------------------------------------------------------------

const SERVER_INSTRUCTIONS: &str = include_str!("../instructions.md");

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Log to stderr (stdout is the MCP transport)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    info!(
        "lumen-sqlite MCP server starting (db_dir={})",
        std::env::var("LUMEN_SQLITE_DB_DIR").unwrap_or_else(|_| ".".into())
    );

    let server = LumenSqlite::new();
    let transport = copilot_compat::CopilotDiscoveryTransport::new(
        rmcp::transport::async_rw::AsyncRwTransport::new_server(
            tokio::io::stdin(),
            tokio::io::stdout(),
        ),
    );
    let ct = server.serve(transport).await?;
    ct.waiting().await?;
    Ok(())
}
