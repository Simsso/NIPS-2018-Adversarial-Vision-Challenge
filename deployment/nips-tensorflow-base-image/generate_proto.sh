echo "Generate protobufs for $1"

protoc -I  . TrainingProto.proto --go_out=plugins=grpc:./