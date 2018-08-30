package Training

import (
	"context"
	"fmt"
)

type trainingManagerServer struct {
	trainingJobs map[string]*TrainingJob;
}

func (s *trainingManagerServer) RegisterTraining(ctx context.Context, trainingJob *TrainingJob) (*Response, error) {
	fmt.Printf("Register %s", trainingJob.TrainingId)

	s.trainingJobs[trainingJob.TrainingId] = trainingJob
	return &Response{Success: true}, nil
}

func (s *trainingManagerServer) UpdateTraining(ctx context.Context, trainingJob *TrainingJob) (*Response, error) {
	fmt.Printf("Update %s", trainingJob.TrainingId)

	s.trainingJobs[trainingJob.TrainingId] = trainingJob
	return &Response{Success: true}, nil
}
