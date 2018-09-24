package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
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
	fmt.Printf("Register %s\n", trainingJob.TrainingId)

	s.trainingJobs[trainingJob.TrainingId] = trainingJob
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) UpdateTraining(ctx context.Context, trainingJob *TrainingProto.TrainingJob) (*TrainingProto.Response, error) {
	fmt.Printf("Update %s\n", trainingJob.TrainingId)

	s.trainingJobs[trainingJob.TrainingId] = trainingJob
	return &TrainingProto.Response{Success: true}, nil
}

func (s *trainingManagerServer) ReceiveEvent(rect *TrainingProto.TrainingJob, stream TrainingProto.TrainingProto_ReceiveEventsServer) error {

	buf := new(bytes.Buffer)
	event := "EVENT"

	for i := 0; i < 100; i++ {
		binary.Write(buf, binary.LittleEndian, i)
		stream.Send(&TrainingProto.Event{ event,  buf.Bytes()  })
	}

	return nil
}
