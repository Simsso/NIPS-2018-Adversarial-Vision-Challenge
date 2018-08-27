variable "instance_name" {
  default = "tensorflow-training-model"
}

variable "model_docker_image" {
  default = "nginx"
}

variable "project" {
  default = "nips-2018-207619"
}

variable "region" {
  default = "europe-west1"
}

variable "zone" {
  default = "europe-west1-b"
}

variable "machine_type" {
  default = "custom-1-5120"
}
