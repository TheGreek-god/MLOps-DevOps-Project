FROM node:22-alpine AS build

WORKDIR /app

COPY f1net-frontend/package.json f1net-frontend/package-lock.json ./
RUN npm ci

COPY f1net-frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]