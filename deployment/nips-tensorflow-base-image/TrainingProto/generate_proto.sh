echo "Generate protobufs .."

protoc -I  . TrainingProto.proto --go_out=plugins=grpc:./