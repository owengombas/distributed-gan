variable "project_id" {}
variable "region" { default = "europe-central2" }
variable "cluster_name" { default = "dgan-cluster" }
variable "gke_node_count" { default = 1 }
variable "docker_image_tag" { default = "amd64" }
variable "docker_image_name" { default = "dgan" }
