uvicorn tapngo.asgi:application --reload
<!-- uvicorn server.asgi:application --reload --port 8000 -->

python manage.py runserver_plus --cert-file cert.pem --key-file key.pem

gcloud auth login
gcloud config set project present-495107
gcloud projects list

gcloud config get-value project

gcloud app deploy app.yaml

gcloud app deploy --quiet app.yaml --promote --stop-previous-version --version main-v1

gcloud app logs tail -s default

gcloud config set project tapngo2

# Build and deploy directly - tapngo2
gcloud run deploy tapngo-qcluster \
  --source . \
  --region europe-west2 \
  --platform managed \
  --min-instances 1 \
  --cpu 1 \
  --memory 512Mi \
  --set-env-vars DJANGO_SETTINGS_MODULE=tapngo.settings,DEBUG=false