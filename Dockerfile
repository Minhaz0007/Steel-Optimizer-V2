# ── Stage 1: Build ─────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app

# Install all deps (including devDependencies for Vite build + tsx)
COPY package*.json ./
RUN npm ci

# Copy source and build the Vite SPA
COPY . .
RUN npm run build

# ── Stage 2: Production runtime ────────────────────────────────────────────
FROM node:20-alpine
WORKDIR /app

ENV NODE_ENV=production
# Cloud Run expects the container to listen on PORT (default 8080)
ENV PORT=8080

# Install only production deps, then add tsx to run server.ts
COPY package*.json ./
RUN npm ci --omit=dev && npm install tsx

# Copy Vite-built frontend and the TypeScript server
COPY --from=builder /app/dist ./dist
COPY server.ts ./

EXPOSE 8080
CMD ["./node_modules/.bin/tsx", "server.ts"]
