# Stage 1: build the static assets
FROM node:24-bookworm-slim AS build

WORKDIR /app

# Install deps first (cached unless package files change)
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: serve the built SPA with nginx (also reverse-proxies /api to the backend)
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
