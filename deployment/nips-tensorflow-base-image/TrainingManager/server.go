package main

import (
	"context"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
)

type trainingManagerServer struct {
	trainingJobs     map[string]*TrainingProto.TrainingJob
	trainingJobsData map[string]*TrainingJobData
}

type TrainingJobData struct {
	taskQueue chan string
	waitForTask chan string
	telegramNotifications chan string
}

func (s *trainingManagerServer) Init() {
	if s.trainingJobs == nil {
		s.trainingJobs = map[string]*TrainingProto.TrainingJob{}
	}

	if s.trainingJobsData == nil {
		s.trainingJobsData = map[string]*TrainingJobData{}
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
	s.trainingJobsData[trainingJob.ModelId].waitForTask <- "UPDATE_DONE"

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) FinishTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob)  (*TrainingProto.Response, error) {
	fmt.Printf("Finish of %s  Reason: %s", trainingJob.ModelId, trainingJob.Status.String())

	// update before shutdown
	s.trainingJobs[trainingJob.ModelId] = trainingJob
	s.trainingJobsData[trainingJob.ModelId].taskQueue <- "SHUTDOWN"

	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) ReceiveEvent(rect *TrainingProto.TrainingJob, stream TrainingProto.TrainingProto_ReceiveEventServer) error {

	for {
		task := <-s.trainingJobsData[rect.ModelId].taskQueue

		if task == "SHUTDOWN" {
			// notify user
			s.trainingJobsData[rect.ModelId].telegramNotifications <- "TRAINING_STOP"

		} else if task == "UPDATE" {
			if err := stream.Send(&TrainingProto.Event{Event: TrainingProto.Event_UPDATE, Data: nil}); err != nil {
				fmt.Printf("Connection closed to %s", rect.ModelId)
				return err
			}
		}
	}

	return nil
}
