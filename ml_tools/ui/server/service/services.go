package service

import (
	"io"
	"ml_tools/server/protocol"
)

type Service interface {
	// OnMessage must be synchronized, so it could be called from
	// multiple go-rutines.
	OnMessage(message protocol.Message)
	GetMessageTypeRange() (min, max int)
}

type Services struct {
	rw      protocol.ReadWriter
	Control *Control
	Video   *Video

	services []Service
}

func NewServies(rw protocol.ReadWriter) *Services {
	s := &Services{}
	s.rw = rw

	s.Control = &Control{rw}
	s.Video = newVideo(rw)

	s.services = []Service{s.Control, s.Video}
	return s
}

func (s *Services) Start() {
	for i := 0; i < 3; i++ {
		go s.startReadingLoop()
	}
}

func (s *Services) startReadingLoop() {
	for {
		message, err := s.rw.Read()
		if err != nil {
			if err == io.EOF {
				return
			}
			panic(err)
		}

		for _, service := range s.services {
			min, max := service.GetMessageTypeRange()
			if int(message.Type) >= min && int(message.Type) <= max {
				service.OnMessage(message)
				break
			}
		}
	}
}
