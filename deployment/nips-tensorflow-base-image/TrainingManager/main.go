package main

import (
	"flag"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"google.golang.org/grpc"
	"log"
	"net"
)

var (
	port = flag.Int("port", 6007, "Port where the gRPC server should listen to. Default: 6007")
)

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	trainingManagerServer := trainingManagerServer{}
	trainingManagerServer.Init()

	TrainingProto.RegisterTrainingProtoServer(grpcServer, &trainingManagerServer)
	grpcServer.Serve(lis)
}
