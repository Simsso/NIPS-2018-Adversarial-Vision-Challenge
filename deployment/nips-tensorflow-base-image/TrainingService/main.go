package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"google.golang.org/grpc"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

var (
	trainingJob       TrainingProto.TrainingJob
	client            TrainingProto.TrainingProtoClient
	serverAddr        string
	port              string
	modelId           string
	scriptPath        string
	logFile           string
	wg                sync.WaitGroup
	connectionTimeout time.Duration
)

func main() {
	flag.StringVar(&serverAddr, "url", "127.0.0.1", "URL to the TrainingManager")
	flag.StringVar(&modelId, "model-id", "no-model", "Name of the trained model")
	flag.StringVar(&port, "port", "6007", "Port of the TrainingManger")
	flag.DurationVar(&connectionTimeout, "connection-timeout", time.Second*5, "Timeout duration to the training manager. Default: 5 Seconds")
	// get current working directory
	dir, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	flag.StringVar(&logFile, "log-file", fmt.Sprintf("%s/log_output.txt", dir), "Path to the log output (bash). Default: log_output.txt in current working directory")
	flag.StringVar(&scriptPath, "script-path", fmt.Sprintf("%s/start.sh", dir), "Path to the script to be executed. Default: start.sh in the current working directory")
	flag.Parse()

	logWithDate("Starting TrainingService ..")

	ctx, _ := context.WithTimeout(context.Background(), 5*time.Second)
	conn, err := grpc.DialContext(ctx, fmt.Sprintf("%s:%s", serverAddr, port), grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatal(err)
		return
	}
	logWithDateAndFormat("Connected as %s to TrainingManager at %s:%s", modelId, serverAddr, port)

	client = TrainingProto.NewTrainingProtoClient(conn)

	// register TrainingJob
	if err := RegisterTrainingJob(time.Now().Unix(), modelId); err != nil {
		log.Fatal(err)
		return
	}

	// start EventListener
	wg.Add(1)
	go EventListener(&client)

	// notify TrainingManager about finished init
	if response, err := client.FinishInit(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't notify Training manager for init of %s", modelId)
		return
	}
	logWithDate("Starting job")
	// start TrainingJob
	if err := StartTraining(); err != nil {
		log.Fatal(err)
		return
	}

	wg.Wait()
	defer conn.Close()
}

func RegisterTrainingJob(startTime int64, modelId string) (error) {

	trainingJob.StartTime = startTime
	trainingJob.ModelId = modelId
	trainingJob.Status = TrainingProto.TrainingJob_INIT

	ReadOuput()

	if response, err := client.RegisterTraining(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't register %s at TrainingManager (%s)", modelId, serverAddr)
		return err
	}

	return nil
}

func EventListener(client *TrainingProto.TrainingProtoClient) {
	logWithDate("Listening for incoming events from the server ..")

	stream, err := (*client).ReceiveEvent(context.Background(), &trainingJob)
	if err != nil {
		log.Fatal(err)
		return
	}

	for {
		event, err := stream.Recv()
		recvEvent := event.Event

		if recvEvent == TrainingProto.Event_UPDATE {
			logWithDate("UPDATE-event received! Sending updates to TrainingManager!")
			UpdateTrainingJob()
		} else if recvEvent == TrainingProto.Event_NVIDIASMI {
			logWithDate("NVIDIASMI-event received! Getting nvidia-smi output and sending it to TrainingManager!")
			SendNVIDIASMI()
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

func StartTraining() (error) {
	logWithDateAndFormat("Starting script at %s", scriptPath)
	cmd := exec.Command("/bin/bash", scriptPath)

	if err := cmd.Start(); err != nil {
		return err
	}

	logWithDate("Script started successfully!")

	// wait for the process to exit and notify TrainingManager
	go waitForProcessToExit(cmd)

	// notify TrainingManager about started process
	ReadOuput()
	trainingJob.Status = TrainingProto.TrainingJob_RUNNING
	if response, err := client.TrainingStart(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't notify Training manager for start of %s", modelId)
		return err
	}

	return nil
}

func waitForProcessToExit(cmd *exec.Cmd) {
	logWithDateAndFormat("Starting aliveness checker for process %d", cmd.Process.Pid)

	if err := cmd.Wait(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			if _, ok := exitError.Sys().(syscall.WaitStatus); ok {
				trainingJob.Status = TrainingProto.TrainingJob_CRASHED
				logWithDateAndFormat("%d has crashed with stderr: %s ", cmd.Process.Pid, exitError.Stderr)
			}
		}
	} else {
		trainingJob.Status = TrainingProto.TrainingJob_FINISHED
		logWithDate("Training exited sucessfully!")
	}

	trainingJob.StopTime = time.Now().Unix()
	ReadOuput()

	if response, err := client.FinishTraining(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't finish %s at TrainingManager (%s)", modelId, serverAddr)
		return
	}
	logWithDate("Waiting for shutdown from TrainingManager ..")

}

func UpdateTrainingJob() (error) {

	ReadOuput()

	if response, err := client.UpdateTraining(context.Background(), &trainingJob); err != nil || !response.Success {
		logWithDateAndFormat("Couldn't update %s at TrainingManager (%s)", modelId, serverAddr)
		return err
	}
	logWithDate("Updating training job at TrainingManager ..")
	return nil
}

func ReadOuput() {
	content, err := ioutil.ReadFile(logFile)
	if err != nil {
		trainingJob.Log = logWithDate("Couldn't find any logs")
	} else {
		logWithDate("Retrieved logfile ..")
		trainingJob.Log = fmt.Sprintf("%s", content)
	}
}

func SendNVIDIASMI() (error){

	return nil
}