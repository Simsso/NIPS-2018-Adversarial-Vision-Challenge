package main

import (
	"context"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
)

type trainingManagerServer struct {
	trainingJobs                map[string]*TrainingProto.TrainingJob
	trainingJobsData            map[string]*TrainingJobData
	telegramNotificationChannel chan telegramNotification
}

type TrainingJobData struct {
	taskQueue   chan string
	waitForTask chan string
}

type telegramNotification struct {
	trainingJob *TrainingProto.TrainingJob
	event       string
}

func (s *trainingManagerServer) Init() {
	if s.trainingJobs == nil {
		s.trainingJobs = map[string]*TrainingProto.TrainingJob{}
	}

	if s.trainingJobsData == nil {
		s.trainingJobsData = map[string]*TrainingJobData{}
	}

	if s.telegramNotificationChannel == nil {
		s.telegramNotificationChannel = make(chan telegramNotification)
	}
}

func (s *trainingManagerServer) RegisterTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Register %s\n", trainingJob.ModelId)

	s.trainingJobs[trainingJob.ModelId] = trainingJob
	s.trainingJobsData[trainingJob.ModelId] = &TrainingJobData{taskQueue: make(chan string), waitForTask: make(chan string)}

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) UpdateTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Update %s\n", trainingJob.ModelId)

	s.trainingJobs[trainingJob.ModelId] = trainingJob

	// mark task as completed
	s.trainingJobsData[trainingJob.ModelId].waitForTask <- "TRAINING_UPDATE_DONE"

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) FinishTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Finish of %s  Reason: %s", trainingJob.ModelId, trainingJob.Status.String())

	// update before shutdown
	s.trainingJobs[trainingJob.ModelId] = trainingJob

	// notify user about shutdown
	s.telegramNotificationChannel <- telegramNotification{trainingJob, "TRAINING_FINISHED"}

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) TrainingStart(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Training Start of %s", trainingJob.ModelId)

	// update
	s.trainingJobs[trainingJob.ModelId] = trainingJob

	// notify user about shutdown
	s.telegramNotificationChannel <- telegramNotification{trainingJob, "TRAINING_STARTED"}

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) FinishInit(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Init of %s has finished", trainingJob.ModelId)

	// update
	s.trainingJobs[trainingJob.ModelId] = trainingJob

	// notify user about shutdown
	s.telegramNotificationChannel <- telegramNotification{trainingJob, "TRAINING_NEW"}

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) ReceiveEvent(rect *TrainingProto.TrainingJob, stream TrainingProto.TrainingProto_ReceiveEventServer) error {

	for {
		task := <-s.trainingJobsData[rect.ModelId].taskQueue

		if task == "UPDATE" {
			if err := stream.Send(&TrainingProto.Event{Event: TrainingProto.Event_UPDATE, Data: nil}); err != nil {
				fmt.Printf("Connection closed to %s", rect.ModelId)
				return err
			}
		}
	}

	return nil
}