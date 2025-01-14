FROM node:23-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable
COPY . /node_modules/fastembed
WORKDIR /node_modules/fastembed

FROM base AS prod-deps
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --prod --frozen-lockfile

FROM base AS build
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile
RUN pnpm run tsc
#RUN pnpm pack 

FROM base
COPY --from=prod-deps /node_modules/fastembed/node_modules/@anush008         /app/node_modules/@anush008
COPY --from=prod-deps /node_modules/fastembed/node_modules/onnxruntime-node  /app/node_modules/onnxruntime-node
COPY --from=prod-deps /node_modules/fastembed/node_modules/progress          /app/node_modules/progress
COPY --from=prod-deps /node_modules/fastembed/node_modules/tar               /app/node_modules/tar
COPY --from=build /node_modules/fastembed                                /app/node_modules/fastembed
