provider "google" {
  credentials = "${file("cloudbuild-service-account.json")}"
  project     = "${var.project}"
  region      = "${var.region}"
}

resource "google_compute_instance" "gce_instance" {
  name         = "${var.instance_name}"
  machine_type = "${var.machine_type}"
  zone         = "${var.zone}"

  boot_disk {
    initialize_params {
      image = "nips-2018-adversarial-visionchallenge-base-image-large"
    }
  }
  
  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  service_account {
    scopes = ["storage-ro"]
  }

   guest_accelerator {
    type = "nvidia-tesla-k80"
    count = 1
  }  
  
  network_interface {
    network       = "default"
    access_config = {}
  }
}

resource "null_resource" "setup-gce" {
  provisioner "local-exec" {
    command = "./configure_gce_instance.sh ${google_compute_instance.gce_instance.network_interface.0.access_config.0.assigned_nat_ip} ${var.model_docker_image}"
  }

  depends_on = ["google_compute_instance.gce_instance"]
}
