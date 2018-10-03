package main

import (
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"google.golang.org/grpc"
	"log"
	"os"
	"time"
)

// For the generation of protobuf:
// protoc -I  . TrainingProto.proto --go_out=plugins=grpc:./TrainingManager/TrainingProto

var trainingJob TrainingProto.TrainingJob
var client TrainingProto.TrainingProtoClient

func main() {
	modelId := os.Getenv("MODEL_ID")
	if len(modelId) == 0 {
		log.Fatalf("no model id specified")
		return
	}

	serverAddr := os.Getenv("TRAININGMANAGER_URL")
	if len(serverAddr) == 0 {
		log.Fatalf("no url of TrainingManager specified")
		return
	}

	conn, err := grpc.Dial(serverAddr, grpc.WithInsecure())
	if err != nil {
		log.Fatal(err)
		return
	}

	trainingJob.StartTime = time.Now().Unix()
	trainingJob.TrainingId = modelId

	client = TrainingProto.NewTrainingProtoClient(conn)

	go EventListener(&client);
	/* response, err := client.RegisterTraining(context.Background(), &trainingJob)
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