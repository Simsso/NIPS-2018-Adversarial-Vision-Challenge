provider "google" {
  credentials = "${file("account.json")}"
  project     = "nips-2018-207619"
  region      = "europe-west4"
}

resource "google_compute_instance" "gce_instance" {
  name         = "test"
  machine_type = "n1-standard-1"
  zone         = "europe-west4-b"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-1604-lts"
    }
  }

/*
  guest_accelerator {
    type = "nvidia-tesla-k80"
    count = 1
  }  
*/
  network_interface {
    network = "default"
    access_config {
    }
  }

}