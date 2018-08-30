package Training

import (
	"flag"
	"fmt"
	"google.golang.org/grpc"
	"log"
	"net"
)

var (
	port    = flag.String("port", "6007", "Port where the gRPC server should listen to. Default: 6007")
	address = flag.String("ip", "localhost", "IP where the gRPC server should bind to. Default: localhost")
)

func Main() {
	flag.Parse()

	lis, err := net.Listen(*address, fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	RegisterTrainingServer(grpcServer, &trainingManagerServer{})
	grpcServer.Serve(lis)
}
