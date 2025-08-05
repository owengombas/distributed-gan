provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_client_config" "default" {}

resource "google_container_cluster" "autopilot" {
  name     = var.cluster_name
  location = var.region

  enable_autopilot = true
  networking_mode  = "VPC_NATIVE"
}

provider "kubernetes" {
  host                   = google_container_cluster.autopilot.endpoint
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.autopilot.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes = {
    host                   = google_container_cluster.autopilot.endpoint
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.autopilot.master_auth[0].cluster_ca_certificate)
  }
}

resource "helm_release" "dgan" {
  name       = "dgan"
  chart      = "${path.module}/../dgan-k8s" # adjust path if needed
  namespace  = "default"
  create_namespace = false

  values = [
    yamlencode({
      replicaCount = 3
      image = {
        repository = "owengombas/dgan"
        tag        = "amd64"
        pullPolicy = "Always"
      }
    })
  ]
}
