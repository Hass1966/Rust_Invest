FROM rust:1.77 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin serve

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/serve /usr/local/bin/rust-invest-serve
COPY --from=builder /app/models/ /app/models/
COPY --from=builder /app/config/ /app/config/
COPY --from=builder /app/rust_invest.db /app/rust_invest.db
WORKDIR /app
EXPOSE 8080
CMD ["rust-invest-serve"]
