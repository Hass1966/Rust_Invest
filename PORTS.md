# Port Assignments

| Port | Service              | Notes                    |
|------|----------------------|--------------------------|
| 8080 | FinOpsMind backend   | Rust/Axum API            |
| 8081 | Rust Invest backend  | Serves frontend + API    |
| 3000 | Nginx → FinOpsMind  | Proxies to 8080          |
| 8082 | Nginx → Rust Invest | Proxies to 8081          |
