package main

import (
	"context"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"time"
)

type trainingManagerServer struct {
	trainingJobs map[string]*TrainingProto.TrainingJob
}

func (s *trainingManagerServer) Init() {
	if s.trainingJobs == nil {
		s.trainingJobs = map[string]*TrainingProto.TrainingJob{}
	}
}

func (s *trainingManagerServer) RegisterTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Register %s\n", trainingJob.ModelId)

	s.trainingJobs[trainingJob.ModelId] = trainingJob
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) UpdateTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Update %s\n", trainingJob.ModelId)

	s.trainingJobs[trainingJob.ModelId] = trainingJob
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) ReceiveEvent(rect *TrainingProto.TrainingJob, stream TrainingProto.TrainingProto_ReceiveEventServer) error {

	// keep the connection alive with the client
	for {
		if err := stream.Send(&TrainingProto.Event{Event: TrainingProto.Event_SHUTDOWN, Data: nil}); err != nil {
			fmt.Printf("Connection closed to %s", rect.ModelId)
			return err
		}
		if err := stream.Send(&TrainingProto.Event{Event: TrainingProto.Event_UPDATE, Data: nil}); err != nil {
			fmt.Printf("Connection closed to %s", rect.ModelId)
			return err
		}
		time.Sleep(time.Second * 20)
	}

	return nil
}
