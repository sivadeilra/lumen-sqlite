use rmcp::RoleServer;
use rmcp::model::{
    ClientJsonRpcMessage, ClientRequest, ErrorCode, ErrorData, JsonRpcError, ServerJsonRpcMessage,
};
use rmcp::transport::Transport;
use tracing::error;

const SERVER_DISCOVER_METHOD: &str = "server/discover";

pub struct CopilotDiscoveryTransport<T> {
    inner: T,
    awaiting_initialize: bool,
}

impl<T> CopilotDiscoveryTransport<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            awaiting_initialize: true,
        }
    }
}

impl<T> Transport<RoleServer> for CopilotDiscoveryTransport<T>
where
    T: Transport<RoleServer>,
{
    type Error = T::Error;

    fn send(
        &mut self,
        item: ServerJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        self.inner.send(item)
    }

    async fn receive(&mut self) -> Option<ClientJsonRpcMessage> {
        loop {
            let message = self.inner.receive().await?;

            match &message {
                ClientJsonRpcMessage::Request(request)
                    if matches!(&request.request, ClientRequest::InitializeRequest(_)) =>
                {
                    self.awaiting_initialize = false;
                    return Some(message);
                }
                ClientJsonRpcMessage::Request(request)
                    if self.awaiting_initialize
                        && matches!(
                            &request.request,
                            ClientRequest::CustomRequest(custom)
                                if custom.method == SERVER_DISCOVER_METHOD
                        ) =>
                {
                    let response = ServerJsonRpcMessage::Error(JsonRpcError::new(
                        request.id.clone(),
                        ErrorData::new(ErrorCode::METHOD_NOT_FOUND, SERVER_DISCOVER_METHOD, None),
                    ));

                    if let Err(send_error) = self.inner.send(response).await {
                        error!(
                            "failed to reject pre-initialize {SERVER_DISCOVER_METHOD} request: \
                             {send_error}"
                        );
                        return None;
                    }
                }
                _ => return Some(message),
            }
        }
    }

    fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::io;
    use std::sync::{Arc, Mutex};

    use rmcp::RoleServer;
    use rmcp::model::{ClientJsonRpcMessage, ErrorCode, ServerJsonRpcMessage};
    use rmcp::transport::Transport;
    use serde_json::json;

    use super::CopilotDiscoveryTransport;

    struct MockTransport {
        incoming: VecDeque<ClientJsonRpcMessage>,
        sent: Arc<Mutex<Vec<ServerJsonRpcMessage>>>,
    }

    impl Transport<RoleServer> for MockTransport {
        type Error = io::Error;

        fn send(
            &mut self,
            item: ServerJsonRpcMessage,
        ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
            let sent = Arc::clone(&self.sent);
            async move {
                sent.lock().unwrap().push(item);
                Ok(())
            }
        }

        fn receive(&mut self) -> impl Future<Output = Option<ClientJsonRpcMessage>> + Send {
            std::future::ready(self.incoming.pop_front())
        }

        fn close(&mut self) -> impl Future<Output = Result<(), Self::Error>> + Send {
            std::future::ready(Ok(()))
        }
    }

    fn message(value: serde_json::Value) -> ClientJsonRpcMessage {
        serde_json::from_value(value).unwrap()
    }

    fn initialize_request(id: i64) -> ClientJsonRpcMessage {
        message(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }))
    }

    fn assert_message_eq(actual: &ClientJsonRpcMessage, expected: &ClientJsonRpcMessage) {
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[tokio::test]
    async fn rejects_pre_initialize_server_discover_and_yields_initialize() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let inner = MockTransport {
            incoming: VecDeque::from([
                message(json!({
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "server/discover",
                    "params": {}
                })),
                initialize_request(1),
            ]),
            sent: Arc::clone(&sent),
        };
        let mut transport = CopilotDiscoveryTransport::new(inner);

        let received = transport.receive().await.unwrap();

        assert_message_eq(&received, &initialize_request(1));
        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 1);
        match &sent[0] {
            ServerJsonRpcMessage::Error(error) => {
                assert_eq!(serde_json::to_value(&error.id).unwrap(), json!(0));
                assert_eq!(error.error.code, ErrorCode::METHOD_NOT_FOUND);
            }
            response => panic!("expected JSON-RPC error response, got {response:?}"),
        }
    }

    #[tokio::test]
    async fn forwards_server_discover_after_initialize() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let discover = message(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "server/discover",
            "params": {}
        }));
        let inner = MockTransport {
            incoming: VecDeque::from([initialize_request(1), discover.clone()]),
            sent: Arc::clone(&sent),
        };
        let mut transport = CopilotDiscoveryTransport::new(inner);

        assert_message_eq(&transport.receive().await.unwrap(), &initialize_request(1));
        assert_message_eq(&transport.receive().await.unwrap(), &discover);
        assert!(sent.lock().unwrap().is_empty());
    }
}
