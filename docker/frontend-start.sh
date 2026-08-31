#!/bin/sh
envsubst '${PREDICT_API_URL} ${INGEST_URL}' < /usr/share/nginx/html/config.js.template > /usr/share/nginx/html/config.js
exec nginx -g 'daemon off;'
