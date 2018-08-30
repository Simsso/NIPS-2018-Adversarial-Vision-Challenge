package Training

import (
	"context"
	"google.golang.org/grpc"
	"log"
	"os"
	"time"
)

// For the generation of protobuf:
// protoc -I  . Training.proto --go_out=plugins=grpc:./TrainingManager

var trainingJob TrainingJob
var client TrainingClient

func Main() {
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

	conn, err := grpc.Dial(serverAddr)
	if err != nil {
		log.Fatal(err)
		return
	}

	trainingJob.StartTime = time.Now().Unix()
	trainingJob.TrainingId = modelId

	client = NewTrainingClient(conn)

	response, err := client.RegisterTraining(context.Background(), &trainingJob)
	if err != nil || !response.Success {
		log.Fatalf("couldn't register %s at TrainingManager", modelId)
		return
	}

	response2, err := client.UpdateTraining(context.Background(), &trainingJob)
	if err != nil || !response2.Success {
		log.Fatalf(" couldn't register %s at TrainingManager", modelId)
		return
	}
	defer conn.Close()
}
