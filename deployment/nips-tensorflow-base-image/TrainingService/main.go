package main

import (
	"flag"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"google.golang.org/grpc"
	"log"
	"time"
)

// For the generation of protobuf:
// protoc -I  . TrainingProto.proto --go_out=plugins=grpc:./TrainingManager/TrainingProto
var (
 trainingJob TrainingProto.TrainingJob
 client TrainingProto.TrainingProtoClient
 serverAddr string
 modelId string
)

func main() {
	flag.StringVar(&serverAddr,"url","127.0.0.1","URL to the training manager")
	flag.StringVar(&modelId, "model_id","no-model", "Name of the trained model")
	flag.Parse()

	logWithDate("Starting ..")
	conn, err := grpc.Dial(serverAddr, grpc.WithInsecure())
	if err != nil {
		log.Fatal(err)
		return
	}

	trainingJob.StartTime = time.Now().Unix()
	trainingJob.TrainingId = modelId

	client = TrainingProto.NewTrainingProtoClient(conn)

	/*go EventListener(&client);
	response, err := client.RegisterTraining(context.Background(), &trainingJob)
	if err != nil || !response.Success {
		log.Fatalf("Couldn't register %s at TrainingManager (%s)", modelId, serverAddr)
		return
	}

	response2, err := client.UpdateTraining(context.Background(), &trainingJob)
	if err != nil || !response2.Success {
		log.Fatalf("Couldn't update %s at TrainingManager (%s)", modelId, serverAddr)
		return
	}
	defer conn.Close() */
}

func EventListener(client *TrainingProto.TrainingProtoClient) {
	fmt.Println("Listening for incoming event from the server ..")


}