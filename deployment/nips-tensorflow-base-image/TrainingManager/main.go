package main

import (
	"flag"
	"fmt"
	"github.com/NIPS-2018-Adversarial-Vision-Challenge/deployment/nips-tensorflow-base-image/TrainingProto"
	"google.golang.org/grpc"
	"gopkg.in/telegram-bot-api.v4"
	"log"
	"net"
	"strings"
	"time"
)

var (
	port = flag.Int("port", 6007, "Port where the gRPC server should listen to. Default: 6007")
)

func main() {
	flag.Parse()

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	trainingManagerServer := trainingManagerServer{}
	trainingManagerServer.Init()

	TrainingProto.RegisterTrainingProtoServer(grpcServer, &trainingManagerServer)

	// Telegram Bot
	go telegramBot(&trainingManagerServer)

	grpcServer.Serve(lis)
}

func telegramBot(trainingManagerServer *trainingManagerServer) {
	bot, err := tgbotapi.NewBotAPI("668078593:AAEwzDMfQiT_Qbaqf5CDO24Mq2s_p6vU_IU")
	if err != nil {
		log.Panic(err)
	}

	bot.Debug = false

	log.Printf("Authorized on account %s", bot.Self.UserName)

	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60

	updates, err := bot.GetUpdatesChan(u)

	var chatIDs []int64

	go telegram_NotificationSender(&chatIDs, bot, trainingManagerServer)

	for update := range updates {
		if update.Message == nil {
			continue
		}
		receivedMessage := update.Message.Text

		if receivedMessage == "/start" {
			chatIDs = append(chatIDs, update.Message.Chat.ID)
			respMessage := tgbotapi.NewMessage(update.Message.Chat.ID, "You were registered for receiving notifications!")
			bot.Send(respMessage)
		} else if receivedMessage == "/list" {
			respMessage := tgbotapi.NewMessage(update.Message.Chat.ID, telegram_cmdList(trainingManagerServer))
			bot.Send(respMessage)
		} else if strings.HasPrefix(receivedMessage, "/shutdown") {
			modelId := strings.Replace(receivedMessage, "/shutdown", "", 1)
			modelId = strings.Trim(modelId, " ")
			respMessage := tgbotapi.NewMessage(update.Message.Chat.ID, telegram_cmdShutdown(trainingManagerServer, modelId))
			bot.Send(respMessage)
		} else if strings.HasPrefix(receivedMessage, "/output") {
			modelId := strings.Replace(receivedMessage, "/output", "", 1)
			modelId = strings.Trim(modelId, " ")
			respMessage := tgbotapi.NewMessage(update.Message.Chat.ID, telegram_cmdOutput(trainingManagerServer, modelId))
			bot.Send(respMessage)
		} else if strings.HasPrefix(receivedMessage, "/nvidia-smi") {
			modelId := strings.Replace(receivedMessage, "/nvidia-smi", "", 1)
			modelId = strings.Trim(modelId, " ")
			respMessage := tgbotapi.NewMessage(update.Message.Chat.ID, telegram_cmdNVIDIASMI(trainingManagerServer, modelId))
			bot.Send(respMessage)
		} else {
			respMessage := tgbotapi.NewMessage(update.Message.Chat.ID, fmt.Sprintf("Command `%s` not available!", receivedMessage))
			bot.Send(respMessage)
		}
	}
}

func telegram_cmdShutdown(trainingManagerServer *trainingManagerServer, modelId string) (message string) {

	if trainingManagerServer.trainingJobs[modelId] == nil {
		message = fmt.Sprintf("Couldn't find training job %s !", modelId)
	} else {
		message = fmt.Sprintf("Shutting down %s", modelId)
		trainingManagerServer.trainingJobsData[modelId].taskQueue <- "SHUTDOWN"
	}

	return
}

func telegram_cmdNVIDIASMI(trainingManagerServer *trainingManagerServer, modelId string) (message string) {
	if trainingManagerServer.trainingJobs[modelId] == nil {
		message = fmt.Sprintf("Couldn't find training job %s !", modelId)
	} else {
		message = fmt.Sprintf("NVIDIA-SMI log of %s:\n", modelId)
		trainingManagerServer.trainingJobsData[modelId].taskQueue <- "NVIDIASMI"
		<-trainingManagerServer.trainingJobsData[modelId].waitForTask
		message += trainingManagerServer.trainingJobs[modelId].NvidiasmiLog
	}

	return
}

func telegram_cmdOutput(trainingManagerServer *trainingManagerServer, modelId string) (message string) {

	if trainingManagerServer.trainingJobs[modelId] == nil {
		message = fmt.Sprintf("Couldn't find training job %s !", modelId)
	} else {
		message = fmt.Sprintf("Log Output from %s:\n", modelId)

		// get recent output
		if trainingManagerServer.trainingJobs[modelId].Status == TrainingProto.TrainingJob_RUNNING {
			trainingManagerServer.trainingJobsData[modelId].taskQueue <- "UPDATE"
			<-trainingManagerServer.trainingJobsData[modelId].waitForTask
		}

		message += trainingManagerServer.trainingJobs[modelId].Log
	}
	return
}

func telegram_cmdList(trainingManagerServer *trainingManagerServer) (message string) {

	message = "Name\t|\tStatus\t|\tStart Date\t|\tStop Date\t\n"
	for trainingJob := range trainingManagerServer.trainingJobs {

		// Get recent information
		trainingManagerServer.trainingJobsData[trainingJob].taskQueue <- "UPDATE"
		<-trainingManagerServer.trainingJobsData[trainingJob].waitForTask

		var trainingStatus string
		var startDate string
		var stopDate string

		if trainingManagerServer.trainingJobs[trainingJob].Status == TrainingProto.TrainingJob_RUNNING {
			trainingStatus = "running"
		} else if trainingManagerServer.trainingJobs[trainingJob].Status == TrainingProto.TrainingJob_CRASHED {
			trainingStatus = "crashed"
		} else if trainingManagerServer.trainingJobs[trainingJob].Status == TrainingProto.TrainingJob_FINISHED {
			trainingStatus = "finished"
		}

		if startUnix := trainingManagerServer.trainingJobs[trainingJob].StartTime; startUnix != 0 {
			startDate = time.Unix(startUnix, 0).Format("15:04:05 2006-01-02")
		} else {
			startDate = "N/A"
		}

		if stopUnix := trainingManagerServer.trainingJobs[trainingJob].StopTime; stopUnix != 0 {
			stopDate = time.Unix(stopUnix, 0).Format("15:04:05 2006-01-02")
		} else {
			stopDate = "N/A"
		}

		message += fmt.Sprintf("%s\t|\t%s\t|\t%s\t|\t%s\n", trainingJob, trainingStatus, startDate, stopDate)
	}

	return
}

func telegram_NotificationSender(chatIDs *[]int64, bot *tgbotapi.BotAPI, trainingManagerServer *trainingManagerServer) {

	for {
		telegramNotification := <-trainingManagerServer.telegramNotificationChannel

		if telegramNotification.event == "TRAINING_NEW" {
			for _, chatID := range *chatIDs {
				messageText := fmt.Sprintf("%s has been initialized!", telegramNotification.trainingJob.ModelId)
				message := tgbotapi.NewMessage(chatID, messageText)
				bot.Send(message)
			}
		} else if telegramNotification.event == "TRAINING_STARTED" {
			for _, chatID := range *chatIDs {
				messageText := fmt.Sprintf("%s has been started!", telegramNotification.trainingJob.ModelId)
				message := tgbotapi.NewMessage(chatID, messageText)
				bot.Send(message)
			}
		} else if telegramNotification.event == "TRAINING_FINISHED" {
			for _, chatID := range *chatIDs {
				var messageText string

				if telegramNotification.trainingJob.Status == TrainingProto.TrainingJob_CRASHED {
					messageText = fmt.Sprintf("%s has crashed! Please issue: /output %s for more information!", telegramNotification.trainingJob.ModelId, telegramNotification.trainingJob.ModelId)
				} else {
					messageText = fmt.Sprintf("%s has finished!", telegramNotification.trainingJob.ModelId)
				}

				message := tgbotapi.NewMessage(chatID, messageText)
				bot.Send(message)
			}
		}
	}
}
