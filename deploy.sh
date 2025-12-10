#!/bin/bash
# ============================================
# Credit Risk API - GCP Cloud Run Deployment Script
# ============================================

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE_NAME="credit-risk-api"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting deployment to Google Cloud Run..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Authenticate (if needed)
echo "📝 Checking authentication..."
gcloud auth print-access-token > /dev/null 2>&1 || gcloud auth login

# Set project
echo "📁 Setting project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable containerregistry.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet

# Build and push image
echo "🏗️  Building and pushing Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} --timeout=20m

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 60 \
    --concurrency 80 \
    --min-instances 0 \
    --max-instances 10

# Get service URL
echo ""
echo "✅ Deployment complete!"
echo ""
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format="value(status.url)")
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📚 API Docs: ${SERVICE_URL}/docs"
echo "❤️  Health Check: ${SERVICE_URL}/health"
