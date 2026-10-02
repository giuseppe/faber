/*
 * faber
 *
 * Copyright (C) 2025 Giuseppe Scrivano <giuseppe@scrivano.org>
 * faber is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * faber is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with faber.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

//! The line-delimited request/response protocol between a faber client
//! (`RemoteDb`) and `faber serve`. faber's own, not JSON-RPC: one JSON
//! object per line each way, every response answering the request with the
//! same `id`.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RpcRequest {
    pub id: u64,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RpcResponse {
    /// The id of the request this answers, or `None` (`null`) if the
    /// request was too malformed to tell.
    pub id: Option<u64>,
    /// `Some(Value::Null)` for a `"result": null` - a method that returns
    /// nothing - and `None` only when the field is missing.
    #[serde(
        default,
        deserialize_with = "present",
        skip_serializing_if = "Option::is_none"
    )]
    pub result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Deserializes a field that's there, even as `null`, to `Some` - serde's
/// default for `Option` would turn an explicit `null` into `None`.
fn present<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer).map(Some)
}

impl RpcResponse {
    pub fn success(id: u64, result: serde_json::Value) -> Self {
        Self {
            id: Some(id),
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<u64>, message: String) -> Self {
        Self {
            id,
            result: None,
            error: Some(message),
        }
    }

    /// The outcome of request `expected_id`, checking that this response
    /// really answers it and carries exactly one of `result` and `error`.
    pub fn into_result(self, expected_id: u64) -> Result<serde_json::Value, String> {
        if self.id != Some(expected_id) {
            return Err(match self.error {
                Some(error) if self.id.is_none() => {
                    format!("server rejected request {}: {}", expected_id, error)
                }
                _ => format!(
                    "response id {:?} doesn't match request id {}",
                    self.id, expected_id
                ),
            });
        }
        match (self.result, self.error) {
            (Some(result), None) => Ok(result),
            (None, Some(error)) => Err(error),
            (None, None) => Err(format!(
                "response {} has neither result nor error",
                expected_id
            )),
            (Some(_), Some(_)) => Err(format!(
                "response {} has both result and error",
                expected_id
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn response(json: serde_json::Value) -> RpcResponse {
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn test_into_result_success_and_error() {
        assert_eq!(
            response(json!({"id": 3, "result": [1]})).into_result(3),
            Ok(json!([1]))
        );
        assert_eq!(
            response(json!({"id": 3, "error": "no such agent"})).into_result(3),
            Err("no such agent".to_string())
        );
        // A method that returns nothing answers with a null result.
        let line = serde_json::to_string(&RpcResponse::success(3, json!(null))).unwrap();
        assert_eq!(line, r#"{"id":3,"result":null}"#);
        let parsed: RpcResponse = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed.into_result(3), Ok(json!(null)));
    }

    #[test]
    fn test_into_result_rejects_mismatched_or_malformed_responses() {
        let err = response(json!({"id": 4, "result": true}))
            .into_result(3)
            .unwrap_err();
        assert!(err.contains("doesn't match"), "{err}");
        let err = response(json!({"id": 3})).into_result(3).unwrap_err();
        assert!(err.contains("neither"), "{err}");
        let err = response(json!({"id": 3, "result": 1, "error": "x"}))
            .into_result(3)
            .unwrap_err();
        assert!(err.contains("both"), "{err}");
        let err = response(json!({"id": null, "error": "invalid request: eof"}))
            .into_result(3)
            .unwrap_err();
        assert_eq!(err, "server rejected request 3: invalid request: eof");
    }

    #[test]
    fn test_unknown_fields_are_rejected() {
        assert!(
            serde_json::from_value::<RpcRequest>(json!({"id": 1, "method": "x", "parameters": {}}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<RpcResponse>(json!({"id": 1, "result": 1, "extra": 2}))
                .is_err()
        );
    }

    #[test]
    fn test_null_id_is_serialized() {
        let line = serde_json::to_string(&RpcResponse::error(None, "bad".into())).unwrap();
        assert_eq!(line, r#"{"id":null,"error":"bad"}"#);
    }
}
