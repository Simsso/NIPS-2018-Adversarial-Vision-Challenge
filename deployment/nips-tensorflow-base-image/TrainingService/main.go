package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"google.golang.org/grpc"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// For the generation of protobuf:
// protoc -I  . TrainingProto.proto --go_out=plugins=grpc:./TrainingManager/TrainingProto
var (
	trainingJob TrainingProto.TrainingJob
	client      TrainingProto.TrainingProtoClient
	serverAddr  string
	port        string
	modelId     string
	scriptPath  string
	wg          sync.WaitGroup
)

func main() {
	flag.StringVar(&serverAddr, "url", "127.0.0.1", "URL to the TrainingManager")
	flag.StringVar(&modelId, "model-id", "no-model", "Name of the trained model")
	flag.StringVar(&port, "port", "6007", "Port of the TrainingManger")

	// get current working directory
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	flag.StringVar(&scriptPath, "script-path", fmt.Sprintf("%s/start.sh", dir), "Path to the script to be executed. Default: start.sh in the current working directory")
	flag.Parse()

	logWithDate("Starting TrainingService ..")

	conn, err := grpc.Dial(fmt.Sprintf("%s:%s", serverAddr, port), grpc.WithInsecure())
	if err != nil {
		log.Fatal(err)
		return
	}
	logWithDateAndFormat("Connected as %s to TrainingManager at %s", modelId, serverAddr)

	client = TrainingProto.NewTrainingProtoClient(conn)

	// start TrainingJob
	if err := StartTraining(); err != nil {
		log.Fatal(err)
		return
	}

	// init TrainingJob
	if err := RegisterTrainingJob(time.Now().Unix(), modelId); err != nil {
		log.Fatal(err)
		return
	}

	// start EventListener
	wg.Add(1)
	go EventListener(&client)
	wg.Wait() // wait until shutdown event received from server

	defer conn.Close()
}

func StartTraining() (error) {
	logWithDateAndFormat("Starting script at %s", scriptPath)
	cmd := exec.Command("/bin/bash", scriptPath)

	if err := cmd.Start(); err != nil {
		return err
	}

	if err := cmd.Wait(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			if status, ok := exitError.Sys().(syscall.WaitStatus); ok {
				return errors.New(fmt.Sprintf("%s exited with %d.\n stderr: %s", scriptPath, status.ExitStatus(), exitError.Stderr))
			}
		}
	}

	logWithDate("Script started successfully!")
	return nil
}

func RegisterTrainingJob(startTime int64, modelId string) (error) {

	trainingJob.StartTime = startTime
	trainingJob.ModelId = modelId
	trainingJob.Running = true

	if response, err := client.RegisterTraining(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't register %s at TrainingManager (%s)", modelId, serverAddr)
		return err
	}

	return nil
}

func UpdateTrainingJob() (error) {

	if response, err := client.UpdateTraining(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't update %s at TrainingManager (%s)", modelId, serverAddr)
		return err
	}

	return nil
}

func EventListener(client *TrainingProto.TrainingProtoClient) {
	logWithDate("Listening for incoming event from the server ..")

	stream, err := (*client).ReceiveEvent(context.Background(), &trainingJob)
	if err != nil {
		log.Fatal(err)
		return
	}

	for {
		event, err := stream.Recv()
		recvEvent := event.Event.String()

		if recvEvent == TrainingProto.Event_UPDATE.String() {
			logWithDate("UPDATE-event received! Sending TrainingJob to TrainingManager!")
			(*client).UpdateTraining(context.Background(), &trainingJob)
		} else if recvEvent == TrainingProto.Event_SHUTDOWN.String() {
			logWithDate("SHUTDOWN-event received!")
		}

		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatalf("%v.EventListener(_) = _, %v", client, err)
			return
		}

	}
	wg.Done()
}
