# Omnivores frontend.
#
# Built to static files and served by nginx. The backend is a separate service,
# reached through the nginx proxy defined in nginx.conf, so the browser talks to
# one origin and CORS never enters the picture.

FROM node:20-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json* ./
# `npm ci` when a lockfile exists (reproducible), `npm install` otherwise.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

COPY . .
RUN npm run build


FROM nginx:alpine AS runtime

COPY --from=builder /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -q --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
