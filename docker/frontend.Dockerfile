FROM node:22-alpine AS build

WORKDIR /app

COPY f1net-frontend/package.json f1net-frontend/package-lock.json ./
RUN npm ci

COPY f1net-frontend/ .
RUN npm run build

FROM nginx:alpine
RUN apk add --no-cache gettext
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
RUN cp /usr/share/nginx/html/config.js /usr/share/nginx/html/config.js.template
COPY docker/frontend-start.sh /docker-entrypoint-custom.sh
RUN chmod +x /docker-entrypoint-custom.sh
EXPOSE 80
CMD ["/docker-entrypoint-custom.sh"]
