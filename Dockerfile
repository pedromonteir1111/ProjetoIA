#Build e Instalação de Dependências
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Copia os arquivos de configuração de dependências
COPY pyproject.toml uv.lock ./

# Instala as dependências criando um ambiente virtual auto-contido
RUN uv sync --frozen --no-install-project --no-dev

FROM python:3.11-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Copia o modelo de IA e o código da aplicação
COPY modelo_mlp_evoluido.keras ./modelo_mlp_evoluido.keras
COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]