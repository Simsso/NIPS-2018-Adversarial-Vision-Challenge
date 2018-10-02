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
	s.trainingJobsData[trainingJob.ModelId] = &TrainingJobData{taskQueue: make(chan string)}
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) UpdateTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Update %s\n", trainingJob.ModelId)

	s.trainingJobs[trainingJob.ModelId] = trainingJob
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) FinishTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	s.trainingJobs[trainingJob.ModelId] = trainingJob
	s.trainingJobsData[trainingJob.ModelId].taskQueue <- "SHUTDOWN"
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) ReceiveEvent(rect *TrainingProto.TrainingJob, stream TrainingProto.TrainingProto_ReceiveEventServer) error {

	for {
		task := <-s.trainingJobsData[rect.ModelId].taskQueue

		if task == "SHUTDOWN" {
			if err := stream.Send(&TrainingProto.Event{Event: TrainingProto.Event_SHUTDOWN, Data: nil}); err != nil {
				fmt.Printf("Connection closed to %s", rect.ModelId)
				return err
			}
		} else if task == "UPDATE" {
			if err := stream.Send(&TrainingProto.Event{Event: TrainingProto.Event_UPDATE, Data: nil}); err != nil {
				fmt.Printf("Connection closed to %s", rect.ModelId)
				return err
			}
		}
	}

	return nil
}
